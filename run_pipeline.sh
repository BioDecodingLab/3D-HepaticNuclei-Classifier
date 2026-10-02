#!/usr/bin/env bash
# Sequential pipeline launcher. Run inside the nuclei_classification environment.
set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${DATA:-$repo_dir/../data}"
results_dir="${RUN:-$repo_dir/../results}"
dino_repo="${DINO_REPO:-$repo_dir/3DINO}"
dino_config="${DINO_CONFIG:-}"
dino_weights="${DINO_WEIGHTS:-}"
python_bin=python
class_names="$repo_dir/config/class_names.json"
workers=8
classical_jobs=1
classical_threads=1
classical_options=()
embedding_batch=16
cnn_batch=256
device=cuda
seed=42
start=1
stop=9
dry_run=0
data_parallel=()

usage() {
    cat <<'HELP'
Usage: bash run_pipeline.sh [options]
  --data DIR             Contains image/, labels/, class/
  --results DIR          Outputs directly under DIR (default: ../results)
  --dino-repo DIR        External 3DINO source checkout
  --dino-config PATH     Matching configuration stem or .yaml file
  --dino-weights FILE    Pretrained weights
  --python EXECUTABLE    Python interpreter (default: python)
  --class-names FILE     Class display metadata (default: config/class_names.json)
  --workers N           Extraction/loading workers (default: 8)
  --classical-jobs N    Concurrent classical searches on this host (default: 1)
  --classical-threads N BLAS/OpenMP threads per search (default: 1)
  --classical-cache DIR Fast local disk for temporary preprocessing caches
  --resume-classical    Resume stage 4 checkpoints from this version
  --embedding-batch N   DINO inference batch size (default: 16)
  --cnn-batch N         CNN training batch size (default: 256)
  --device DEVICE      PyTorch device (default: cuda)
  --seed N             Experiment seed (default: 42)
  --data-parallel      Use multiple visible GPUs for CNN training
  --start-at N         First command stage (default: 1)
  --stop-after N       Last command stage (default: 9)
  --dry-run            Print commands without creating outputs or running Python
  --help               Show this help

Command stages (script numbers are unchanged):
  1 preprocessing       2 features          3 CV assembly
  4 classical select    5 CNN select        6 classical evaluate
  7 CNN evaluate        8 statistics        9 representations

All six augmentation levels and all five folds are used. Start-at continues from
completed prerequisite stages. --resume-classical resumes stage 4 checkpoints
written by this version. Other partial stages still need archival or a new location.
HELP
}

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        --resume-classical) classical_options+=(--resume); shift ;;
        --classical-cache)
            (($# >= 2)) || { echo "Missing cache directory" >&2; exit 2; }
            classical_options+=(--cache-dir "$2"); shift 2 ;;
        --data-parallel) data_parallel=(--data-parallel); shift ;;
        --data|--results|--dino-repo|--dino-config|--dino-weights|--python|--class-names|--workers|--embedding-batch|--cnn-batch|--device|--seed|--start-at|--stop-after|--classical-jobs|--classical-threads)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --data) data_dir=$2 ;;
                --results) results_dir=$2 ;;
                --dino-repo) dino_repo=$2 ;;
                --dino-config) dino_config=$2 ;;
                --dino-weights) dino_weights=$2 ;;
                --python) python_bin=$2 ;;
                --class-names) class_names=$2 ;;
                --workers) workers=$2 ;;
                --classical-jobs) classical_jobs=$2 ;;
                --classical-threads) classical_threads=$2 ;;
                --embedding-batch) embedding_batch=$2 ;;
                --cnn-batch) cnn_batch=$2 ;;
                --device) device=$2 ;;
                --seed) seed=$2 ;;
                --start-at) start=$2 ;;
                --stop-after) stop=$2 ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
dino_config="${dino_config:-$dino_repo/dinov2/configs/train/vit3d_highres}"
dino_weights="${dino_weights:-$data_dir/3dino_vit_weights.pth}"
for value in "$classical_jobs" "$classical_threads" "$workers" "$embedding_batch" "$cnn_batch" "$seed" "$start" "$stop"; do
    [[ $value =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Expected nonnegative integer: $value" >&2; exit 2; }
done
((classical_jobs > 0 && classical_threads > 0 && embedding_batch > 0 && cnn_batch > 0 && start >= 1 && stop <= 9 && start <= stop)) || {
    echo "Invalid batch size or stage range" >&2; exit 2;
}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if (( ! dry_run )); then
    command -v "$python_bin" >/dev/null
    command -v flock >/dev/null || { echo "The Linux flock utility is required" >&2; exit 2; }
    mkdir -p "$results_dir/logs"
    exec 9>"$results_dir/.pipeline.lock"
    flock -n 9 || { echo "Another runner is using $results_dir" >&2; exit 2; }
    preflight=("$python_bin" "$repo_dir/scripts/check_environment.py")
    # GPU resources are checked for extraction or CNN stages.
    if ((start <= 2 && stop >= 2 || start <= 5 && stop >= 5 || start <= 7 && stop >= 7)); then
        preflight+=(--device "$device")
    fi
    if ((start <= 2 && stop >= 2)); then
        preflight+=(--dino-repo "$dino_repo"
            --dino-config "$dino_config" --dino-weights "$dino_weights")
    fi
    "${preflight[@]}"
    "$python_bin" -m pip freeze > "$results_dir/logs/environment_$(date -u +%Y%m%dT%H%M%SZ).txt"
fi

run_stage() {
    local number=$1 name=$2
    shift 2
    ((number >= start && number <= stop)) || return 0
    printf '\nStage %s: %s\n' "$number" "$name"
    printf '%q ' "$@"
    printf '\n'
    (( ! dry_run )) || return 0
    # Both searches must finish before any test evaluation begins.
    if ((number == 6 || number == 7)); then
        for selection in "$results_dir/classical/selection.json" "$results_dir/resnet/selection.json"; do
            [[ -s $selection ]] || { echo "Missing completed selection: $selection" >&2; return 1; }
        done
    fi
    local logfile="$results_dir/logs/${number}_${name}_$(date -u +%Y%m%dT%H%M%SZ).log"
    if "$@" 2>&1 | tee "$logfile"; then
        printf 'Completed stage %s: %s\n' "$number" "$name"
    else
        local status=$?
        echo "Stage $number failed (exit $status). See $logfile. Later stages were not run." >&2
        return "$status"
    fi
}

run_stage 1 preprocessing "$python_bin" "$repo_dir/scripts/1_preprocessing.py" \
    --images "$data_dir/image" --instances "$data_dir/labels" --classes "$data_dir/class" \
    --output "$results_dir/preprocessed" --seed "$seed" --min-voxels 2500 --border-margin 3 \
    --class-names "$class_names"
run_stage 2 features "$python_bin" "$repo_dir/scripts/2_embedding_extraction.py" \
    --data "$results_dir/preprocessed" --output "$results_dir/features" \
    --families dino handcrafted --levels 100 200 500 1000 2000 4000 \
    --dino-repo "$dino_repo" --dino-config "$dino_config" --dino-weights "$dino_weights" \
    --device "$device" --batch-size "$embedding_batch" --workers "$workers" --seed "$seed"
run_stage 3 cv "$python_bin" "$repo_dir/scripts/3_cross_validation_data.py" \
    --data "$results_dir/preprocessed" --features "$results_dir/features" --output "$results_dir/cv"
run_stage 4 classical-select "$python_bin" "$repo_dir/scripts/4_run_models.py" \
    --phase select --cv "$results_dir/cv" --output "$results_dir/classical" --seed "$seed" \
    --jobs "$classical_jobs" --threads-per-job "$classical_threads" "${classical_options[@]}"
run_stage 5 cnn-select "$python_bin" "$repo_dir/scripts/5_run_cNN.py" \
    --phase select --data "$results_dir/preprocessed" --output "$results_dir/resnet" \
    --model resnet3d_18 --device "$device" --batch-size "$cnn_batch" --workers "$workers" \
    --seed "$seed" "${data_parallel[@]}"
run_stage 6 classical-evaluate "$python_bin" "$repo_dir/scripts/4_run_models.py" \
    --phase evaluate --cv "$results_dir/cv" --output "$results_dir/classical"
run_stage 7 cnn-evaluate "$python_bin" "$repo_dir/scripts/5_run_cNN.py" \
    --phase evaluate --data "$results_dir/preprocessed" --output "$results_dir/resnet" \
    --device "$device" --workers "$workers" --seed "$seed"
run_stage 8 statistics "$python_bin" "$repo_dir/scripts/6_statistical_analysis.py" \
    --results "$results_dir/classical" "$results_dir/resnet" --output "$results_dir/statistics" \
    --all-pairwise --seed "$seed"
run_stage 9 representations "$python_bin" "$repo_dir/scripts/7_representation_analysis.py" \
    --features "$results_dir/features" --output "$results_dir/representations" \
    --seed "$seed" --neighbors 50 --min-dist 0.5
printf '\nRequested stages finished successfully. Results: %s\n' "$results_dir"
