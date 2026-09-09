# 3D Hepatic Nuclear Morphotype Classification

A Python workflow for classifying hepatic nuclear morphotypes from **3D DAPI microscopy images**. The repository compares three approaches:

| Approach | Input representation | Classifier |
|---|---|---|
| **3DINO embeddings** | 1,024-dimensional features from a frozen pretrained 3DINO model | Logistic Regression, Random Forest, SVM or MLP |
| **Handcrafted nuclear features** | 31 descriptors of morphology, intensity, texture and complexity | Logistic Regression, Random Forest, SVM or MLP |
| **Direct 3D classification** | Standardized DAPI nuclear patches | ResNet3D-18-style CNN |

The workflow includes nuclear patch extraction, training augmentation, feature extraction, model selection, held-out evaluation, statistical comparisons and exploratory visualization. All classification inputs are DAPI-derived; reference class maps provide the labels.

## Contents

- [Experimental design](#experimental-design)
- [Installation](#installation)
- [Input data and paths](#input-data-and-paths)
- [Running the pipeline](#running-the-pipeline)
- [Run the complete workflow](#run-the-complete-workflow)
- [Continue from stage 4](#continue-from-stage-4)
- [Classical parallelism and resource use](#classical-parallelism-and-resource-use)
- [Resume classical selection](#resume-classical-selection)
- [Preprocessing and features](#preprocessing-and-features)
- [Figures and tables](#figures-and-tables)
- [Running jobs in the background](#running-jobs-in-the-background)
- [Reproducibility](#reproducibility)
- [Repository structure](#repository-structure)
- [Data and references](#data-and-references)

## Experimental design

The dataset contains **five mouse liver volumes, each from a different animal**, with isotropic source voxels of **0.3 × 0.3 × 0.3 µm**. Five nuclear classes are represented:

| Class code | Nuclear morphotype |
|---|---|
| 1 | Hepatocyte |
| 2 | Stellate cell |
| 3 | Kupffer cell |
| 4 | Endothelial cell |
| 5 | Other cell |

Display names are defined externally in [config/class_names.json](config/class_names.json). Pass a different JSON mapping with `--class-names FILE` to preprocessing or to `run_pipeline.sh`. The supplied runner uses this hepatic mapping by default. The Python scripts default to generic “Class 1” through “Class 5” when no mapping is provided; the current model configuration requires five classes.

Preprocessing records the mapping in `manifest.json`. All subsequent stages inherit it for QC plots, confusion matrices, ROC curves, PCA/UMAP legends and classification reports. Model comparisons reject conflicting mappings. Numeric class IDs, probability columns and patch folders such as `label_1` remain stable. Define names before starting a run: do not edit a completed manifest, because its checksum links downstream artifacts to their inputs. When continuing with `--start-at`, the saved mapping is used.

Evaluation uses **five-fold leave-one-animal-out cross-validation**. In each fold, one animal is reserved for testing. Original nuclei from the other four animals are split into training and validation, with **10% of the development nuclei reserved for validation**, stratified jointly by animal and class. The validation count is rounded up to the next integer.

These splits are saved during preprocessing and shared by every model and augmentation condition. **Only training nuclei are augmented.** Validation and test nuclei remain unaugmented, and augmented copies of a nucleus cannot cross subsets within a fold.

Validation macro-F1 determines model selection, with balanced accuracy and lower log-loss used to break ties. For feature-based classifiers, validation also selects the augmentation level. For the CNN, it selects hyperparameters and training checkpoints; the CNN does not select among the six cached augmentation levels. The selected fitted model is evaluated on the held-out animal without a train-plus-validation refit. Selection is performed independently in each fold, so the selected configuration can differ between folds.

The primary comparison is between the three validation-selected approaches, using their held-out animal scores. Test scores are not used to select classifiers, augmentation counts or checkpoints. Exploratory PCA/UMAP and correlation analyses are kept separate from predictive model fitting.

## Installation

### 1. Activate the Python environment

The consolidated `requirements.txt` targets **Python 3.12 on Linux x86_64**, with **glibc 2.28 or newer** and an NVIDIA driver supporting the CUDA 12.4 PyTorch wheels. Run installation commands from the repository root. This file includes the pipeline and 3DINO runtime dependencies.

If the environment has not yet been created:

```bash
conda create -n nuclei_classification python=3.12 pip -y
```

Activate it in each new terminal session:

```bash
conda activate nuclei_classification
export PYTHONNOUSERSITE=1
python --version
```

Confirm that the output is `Python 3.12.x` before installing dependencies. If environment creation or activation fails, resolve that error before continuing.

### 2. Install Python dependencies and PyTorch

```bash
python -m pip install --upgrade pip &&
python -m pip install -r requirements.txt &&
python -m pip check
```

The file pins PyTorch 2.6.0, torchvision 0.21.0 and xFormers 0.0.29.post3 to compatible CUDA 12.4 builds. MONAI 1.5.1 and NiBabel 5.3.2 support the numerical stack. The CPU-based PCA/UMAP and classifiers do not require cuML. See [Dependency specification](docs/DEPENDENCIES.md) for the supported platform, exact dependency choices and validation scope.

### 3. Install 3DINO and obtain the pretrained weights

**3DINO (also referred to here as DINO3D) must be installed separately before embedding extraction.** Its source code and pretrained weights are not included in this repository.

- Obtain the source from the [official 3DINO repository](https://github.com/AICONSlab/3DINO), or use your existing compatible checkout.
- Install this repository's consolidated requirements inside `nuclei_classification`; do not subsequently install the upstream CUDA 11 requirements into that environment.
- Obtain the [pretrained 3DINO-ViT weights](https://huggingface.co/AICONSlab/3DINO-ViT) and their matching configuration.
- Set `DINO_REPO`, `DINO_CONFIG` and `DINO_WEIGHTS` as described below.

The loader uses `dinov2.configs.load_and_merge_config_3d` and `dinov2.eval.setup.build_model_for_eval`. It imports the source directly from `--dino-repo`; a separate editable installation is not necessary. Use matching weights/configuration; the expected output is one 1,024-dimensional vector per patch. Configuration paths may include `.yaml` or omit it. Record the checkout revision used for each experiment.

Check the numerical environment:

```bash
python -m pip check
python -c "import numpy, skimage, torch; print('NumPy:', numpy.__version__); print('scikit-image:', skimage.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Input data and paths

Each animal requires three matching 3D TIFF volumes:

| Folder | Contents |
|---|---|
| `data/image/` | DAPI intensity volumes |
| `data/labels/` | Nuclear instance labels: background `0`, a unique positive integer for each nucleus |
| `data/class/` | Reference class maps: background/unlabelled `0`, class codes `1–5` |

Matching files must have the same filename stem and array shape, for example `image/1.tif`, `labels/1.tif` and `class/1.tif`. Both `.tif` and `.tiff` are supported. This setup assumes **one image per animal**; images from the same animal must not be treated as independent animals.

The commands below use the project server layout. Change the paths for your own installation and run this setup block in each new terminal session:

```bash
conda activate nuclei_classification
export PYTHONNOUSERSITE=1

PROJECT=/medicina/hmorales/projects/Nuclei3DClassification
DATA="$PROJECT/data"
RUN="$PROJECT/results"
DINO_REPO="$PROJECT/code/3DINO"
DINO_CONFIG="$DINO_REPO/dinov2/configs/train/vit3d_highres"
DINO_WEIGHTS="$DATA/3dino_vit_weights.pth"

cd "$PROJECT/code"
mkdir -p "$RUN"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`PROJECT/code` should contain this repository's `scripts/`, `docs/`, `requirements.txt` and `run_pipeline.sh`.

Before starting a full experiment, check the installed packages, DINO configuration and one synthetic forward pass:

```bash
python scripts/check_environment.py --device cuda \
  --dino-repo "$DINO_REPO" --dino-config "$DINO_CONFIG" \
  --dino-weights "$DINO_WEIGHTS" --forward-check
```

This verifies that your checkpoint can produce finite 1,024-dimensional outputs on your GPU. It does not train a model or use biological data.

## Run the complete workflow

After activating the environment and defining the paths above:

```bash
bash run_pipeline.sh \
  --data "$DATA" --results "$RUN" \
  --dino-repo "$DINO_REPO" --dino-config "$DINO_CONFIG" \
  --dino-weights "$DINO_WEIGHTS" \
  --classical-jobs 8 --classical-threads 1 \
  --workers 8 --embedding-batch 16 --cnn-batch 256 --data-parallel
```

Results are saved directly in `$PROJECT/results/`, with subfolders for each stage. The runner uses all five folds and all six augmentation levels, checks the environment, writes individual command logs in `results/logs/`, and stops immediately if a command fails. Both selection stages must finish before either test evaluation stage runs. A lock prevents two runner instances from using the same results folder simultaneously.

To run the whole workflow after closing PuTTY:

```bash
nohup bash run_pipeline.sh \
  --data "$DATA" --results "$RUN" \
  --dino-repo "$DINO_REPO" --dino-config "$DINO_CONFIG" \
  --dino-weights "$DINO_WEIGHTS" \
  --classical-jobs 8 --classical-threads 1 \
  --workers 8 --embedding-batch 16 --cnn-batch 256 --data-parallel \
  > "$RUN/pipeline.log" 2>&1 < /dev/null &
echo $! > "$RUN/pipeline.pid"
```

Preview commands with `bash run_pipeline.sh --dry-run`. The runner has nine command stages because selection and evaluation are separate:

| Runner stage | Command |
|---|---|
| 1 | Script 1: preprocessing |
| 2 | Script 2: feature extraction |
| 3 | Script 3: CV assembly |
| 4 | Script 4: classical model selection |
| 5 | Script 5: CNN selection |
| 6 | Script 4: classical evaluation |
| 7 | Script 5: CNN evaluation |
| 8 | Script 6: statistics |
| 9 | Script 7: representations |

If preprocessing and feature extraction have already completed, add `--start-at 3` to continue with CV assembly. `--stop-after N` ends after a specified stage. These options require complete prerequisite outputs. Classical selection additionally supports candidate-level resume with `--resume-classical`; other stages do not gain resume support from this option. Completed compatible outputs can be kept at their existing path by passing `--results` explicitly. See `bash run_pipeline.sh --help` for options.

## Continue from stage 4

When `preprocessed/`, `features/` and `cv/` are complete, keep the same `RUN` directory and skip stages 1–3. From the repository root, after activating the environment and defining `DATA` and `RUN`:

```bash
nohup bash run_pipeline.sh \
  --data "$DATA" --results "$RUN" \
  --start-at 4 --stop-after 9 \
  --classical-jobs 8 --classical-threads 1 \
  --workers 8 --cnn-batch 256 --data-parallel \
  > "$RUN/pipeline_from4.log" 2>&1 < /dev/null &
echo $! > "$RUN/pipeline_from4.pid"
```

This runs classical selection, CNN selection, both evaluations, statistics and representation analysis in sequence. Use a new or empty `classical/` directory for a fresh search; see [Resume classical selection](#resume-classical-selection) for an interrupted search. Later stages must also satisfy their existing output-directory requirements.

```bash
tail -f "$RUN/pipeline_from4.log"
```

The runner does not wait for or reserve a free GPU. GPU stages require available resources, and the runner performs a CUDA availability check before executing a stage range that includes them. To run classical selection alone while GPU work is deferred, use `--start-at 4 --stop-after 4`. After it completes and GPU resources are available, continue with `--start-at 5 --stop-after 9` and the same data, results and CNN settings. On a managed cluster, request resources through the local scheduler.

## Classical parallelism and resource use

Classical selection parallelizes independent **feature-family / fold / augmentation-level / classifier** searches on one host. Large SVM augmentation levels are scheduled first across folds. Each worker processes its hyperparameter candidates in their original order; the final selection uses the original ordering for exact ties, regardless of execution order.

| Runner option | Default | Effect |
|---|---|---|
| `--classical-jobs N` | `1` | Number of concurrent classical condition searches |
| `--classical-threads N` | `1` | BLAS/OpenMP thread limit inside each classical worker |
| `--classical-cache DIR` | Within `classical/.preprocessing_cache/` | Location for temporary fitted preprocessing and transformed training arrays |
| `--resume-classical` | Off | Resume a compatible classical search or verify a completed selection |
| `--workers N` | `8` | Feature-extraction and CNN data-loading workers; independent of classical search parallelism |

The examples explicitly request **eight searches with one numerical thread each**. Defaults remain one search and one thread when these options are omitted. Keep `--classical-threads 1` initially: increasing it can help numerical operations such as PCA but does not make an individual SVM fit multithreaded. Eight jobs with four threads permit up to 32 numerical threads and are not necessarily faster than eight single-threaded jobs.

For a server with 48 CPU cores, increase `--classical-jobs` only after checking RAM, swap activity and cache-disk use. Each worker loads its own condition and creates additional arrays for preprocessing and fitting. An 80,000 × 1,024 float32 matrix alone occupies 312.5 MiB; total worker memory is substantially larger. There is no automatic RAM or disk-space budget. The requested jobs × threads must not exceed the CPU count available to the process as reported by joblib, which may be smaller than the physical server under a scheduler or container quota. This implementation does not distribute work across separate cluster nodes.

```bash
nproc
free -h
df -h "$RUN"
```

Repeated training-fitted scaler/PCA calculations are cached within each condition. SVM calibration keeps separate training subsets and preprocessing fits for its identity-isolated internal folds. Hyperparameter grids, PCA solver, calibration, validation selection and training data remain unchanged. Classifiers themselves are still fitted separately for every candidate.

Temporary caches can occupy several GB per worker. Prefer a fast local SSD with sufficient space; for example, add `--classical-cache /path/to/local/scratch/nuclei_cache`. Caches are removed for each successfully completed condition, and saved models remain usable without them. Avoid RAM-backed cache directories unless their memory use is explicitly budgeted. Cache files contain training representations and should use the same access restrictions as the input data.

For direct invocation of `scripts/4_run_models.py`, the corresponding options are `--jobs`, `--threads-per-job`, `--cache-dir` and `--resume`. Direct invocation additionally supports `--no-cache`. These settings affect classical selection, not CNN training or evaluation parallelism.

## Resume classical selection

The selector checkpoints after every completed candidate. Each checkpoint contains the completed search prefix and the best model so far. A candidate interrupted during fitting must run again; previously checkpointed candidates and completed conditions are reused. `START` and `DONE` log messages report candidate progress, elapsed time and validation macro-F1.

To resume an interrupted stage 4 and then run stages 5–9, use:

```bash
nohup bash run_pipeline.sh \
  --data "$DATA" --results "$RUN" \
  --start-at 4 --stop-after 9 --resume-classical \
  --classical-jobs 8 --classical-threads 1 \
  --workers 8 --cnn-batch 256 --data-parallel \
  > "$RUN/pipeline_from4_resume.log" 2>&1 < /dev/null &
echo $! > "$RUN/pipeline_from4_resume.pid"
```

Keep the same inputs, seed, model grids, software environment, selection code and threads-per-job. Resume verifies these against `classical/search_config.json` and refuses incompatible changes. The number of concurrent jobs may change. If a custom cache directory was used, pass it again to retain preprocessing-cache reuse. Do not modify the CV files during a search.

A completed `selection.json` is verified and left unchanged. This option does **not** resume a partially trained CNN or skip later completed stages. If stage 4 is already complete and a later stage failed, choose the appropriate `--start-at` value and satisfy that stage's output-directory requirements.

### Existing output without checkpoints

An interrupted classical search created by an implementation without `search_config.json` and candidate checkpoints cannot be imported automatically. Stop that job and confirm its child processes have exited before replacing scripts or moving its output. Archive only the partial classical output, retaining completed preprocessing, features and CV data:

```bash
if [ -d "$RUN/classical" ]; then
  mv -- "$RUN/classical" "$RUN/classical_before_parallel_$(date -u +%Y%m%dT%H%M%SZ)"
fi
```

Then start a fresh stage-4 search. The archived models remain available, but this new search recomputes them. Do not archive a compatible checkpointed search that you intend to resume.

## Running the pipeline

Run the numbered steps in order. Complete **both model-selection stages before starting final evaluation**. Each script provides `--help` for its full command-line options.

### 1. Extract nuclear patches and define the splits

```bash
python scripts/1_preprocessing.py \
  --images "$DATA/image" --instances "$DATA/labels" --classes "$DATA/class" \
  --output "$RUN/preprocessed" \
  --class-names config/class_names.json \
  --seed 42 --min-voxels 2500 --border-margin 3
```

This step checks the input volumes, excludes undersized and border-touching nuclei, normalizes DAPI intensities, and saves each nuclear patch with its original instance mask. It also generates size/count plots, QC tables and the shared train/validation/test split manifest.

**Output:** `$RUN/preprocessed`

### 2. Extract 3DINO embeddings and handcrafted features

```bash
python scripts/2_embedding_extraction.py \
  --data "$RUN/preprocessed" --output "$RUN/features" \
  --families dino handcrafted --levels 100 200 500 1000 2000 4000 \
  --dino-repo "$DINO_REPO" --dino-config "$DINO_CONFIG" \
  --dino-weights "$DINO_WEIGHTS" \
  --device cuda --batch-size 16 --workers 8
```

Both representations are extracted from the same standardized patches and paired augmentation plans. The 3DINO backbone remains frozen and runs in evaluation mode.

Each augmentation target specifies the **total sampled training patches per class per training animal**. At a target of 4,000, a fold contains `4 animals × 5 classes × 4,000 = 80,000` augmented training rows. These are sampled with replacement from training nuclei; they are not 80,000 distinct original nuclei.

The six levels use nested sampling plans. The largest level is extracted once per fold and representation, then subsetted to create smaller conditions. Original, unaugmented features are saved separately for validation, testing, the original-only baseline and exploratory analysis.

**Output:** `$RUN/features`

### 3. Assemble cross-validation datasets

```bash
python scripts/3_cross_validation_data.py \
  --data "$RUN/preprocessed" --features "$RUN/features" \
  --output "$RUN/cv"
```

This step assembles all five folds for both feature families, including the **original-only baseline** and every requested augmentation level. It uses the splits saved in step 1 and checks sample identities, labels and feature schemas.

**Output:** `$RUN/cv`

### 4. Train and select feature-based classifiers

```bash
python scripts/4_run_models.py --phase select \
  --cv "$RUN/cv" --output "$RUN/classical" \
  --jobs 8 --threads-per-job 1
```

Logistic Regression, Random Forest, SVM and MLP are fitted for each feature family, fold and augmentation condition. Feature scaling and predictive PCA are fitted on training data only. MLP early stopping uses the shared validation set. SVM probability calibration uses internal training folds grouped by original nucleus identity.

Validation scores select the fitted models, including the classifier and augmentation level for each representation family. Search results and the selected model identities are saved before test evaluation. Candidate checkpoints are committed during the search; `selection.json` is written only after every requested condition completes.

**Output:** `$RUN/classical`

### 5. Train and select the direct 3D CNN

```bash
python scripts/5_run_cNN.py --phase select \
  --data "$RUN/preprocessed" --output "$RUN/resnet" \
  --model resnet3d_18 --device cuda \
  --batch-size 256 --workers 8 --data-parallel
```

The CNN uses the same validation identities as the feature-based models. Training uses weighted cross-entropy, AdamW, a learning-rate scheduler, mixed precision on CUDA and seeded augmentation that varies by epoch. The best validation checkpoint is saved.

The direct CNN uses online augmentation of original training nuclei rather than the six finite resampling conditions. The default training budget is 100 epochs with early-stopping patience 5. If GPU memory requires a smaller batch size, set `--batch-size` explicitly and record it with the experiment.

An optional 3DINO classification-head experiment is available through `--model dino_cls`, together with `--dino-repo`, `--dino-config` and `--dino-weights`. Use a separate output folder for that experiment and provide the DINO arguments during its evaluation as well.

**Output:** `$RUN/resnet`

### Final evaluation

After steps 4 and 5 have finished and the experimental choices are fixed, evaluate the saved models:

```bash
python scripts/4_run_models.py --phase evaluate \
  --cv "$RUN/cv" --output "$RUN/classical"

python scripts/5_run_cNN.py --phase evaluate \
  --data "$RUN/preprocessed" --output "$RUN/resnet" \
  --device cuda --workers 8
```

Evaluation loads the saved models/checkpoints and exports predictions, metrics, confusion matrices and ROC curves. The primary comparison uses the configuration already selected by validation in each fold. Results for all fixed classifier/augmentation conditions are also exported for descriptive comparisons.

### 6. Compare models and generate statistical summaries

```bash
python scripts/6_statistical_analysis.py \
  --results "$RUN/classical" "$RUN/resnet" \
  --output "$RUN/statistics" --all-pairwise
```

The main plots compare the validation-selected **3DINO**, **handcrafted-feature** and **CNN** approaches on the same five held-out animals. They show individual animal scores and pairwise brackets labelled with Holm-adjusted Wilcoxon p-values.

Macro-F1 is the primary metric; weighted-F1 and balanced accuracy are complementary. Tables include paired effect sizes, model ranks and a within-animal permutation test of the Friedman rank statistic. Holm correction covers the tested pairs across all three comparison metrics within each exported comparison family.

Animal-level means give each animal equal weight. Pooled confusion matrices and ROC curves summarize individual nuclear predictions and therefore weight animals by their numbers of nuclei. With only five animals and overlapping CV training sets, inferential statistics should be interpreted as exploratory.

**Output:** `$RUN/statistics`

### 7. Visualize representations and feature correlations

```bash
python scripts/7_representation_analysis.py \
  --features "$RUN/features" --output "$RUN/representations" \
  --seed 42 --neighbors 50 --min-dist 0.5
```

This analysis uses **original, unaugmented nuclei**, matched by identity across the two feature families. It generates:

- **PCA and UMAP panels:** class- and animal-labelled representation plots, PCA component pairs, scree plots, coordinates and loadings.
- **Corrrelation PCA-3DINO and features:** correlations between the first 31 principal components of standardized 3DINO embeddings and the 31 handcrafted features. PC labels include their explained variance.
- **Corrrelation 3DINO and features:** correlations between the 31 highest-variance individual 3DINO embedding dimensions and the same handcrafted features. Dimensions are ranked by variance before standardization.
- **Supplementary outputs:** Spearman heatmaps and correlation tables stratified by animal and class.

These analyses describe representation structure and do not feed into model selection. Constant-feature correlations are displayed as undefined rather than assigned a numerical association.

**Output:** `$RUN/representations`

## Preprocessing and features

DAPI volumes are normalized independently using the **30th and 99.999th percentiles** and clipped to `[0, 1]`. Nuclear class labels are assigned by the majority nonzero class within each instance mask. Ties and exclusions are recorded in QC tables.

Each extracted patch is min–max normalized, center-padded to **70³ voxels**, or to a larger enclosing cube when necessary, and resized to **112³ voxels**. Images use `order=0` interpolation with antialiasing; masks use matching nearest-neighbor resizing without antialiasing. The transformed original instance mask defines the nucleus throughout feature extraction. DINO/CNN inputs are mapped to `[-1, 1]`; handcrafted intensity features use `[0, 1]`.

The 31 handcrafted features describe morphology, intensity, full 3D masked texture and finite-scale complexity. Mean radius is measured from the nuclear centroid to the surface, and shape index is estimated from local principal curvatures. **[Feature definitions](docs/FEATURES.md)** provides the complete 31-column specification, formulas, units, texture settings and interpretation conventions. The file is included at `docs/FEATURES.md` in this repository.

**Geometric descriptors are measured in standardized voxel coordinates.** They should not be interpreted as original physical dimensions by assigning 0.3 µm to each resized voxel. Original physical nuclear volume is exported separately during preprocessing.

Training augmentation includes axis permutations, independent flips (`p=0.3`), intensity scaling `0.9–1.1` and shifting `−0.05–0.05` (joint `p=0.3`), Gaussian blur with sigma `0.4–1.2` (`p=0.3`), and Gaussian noise with sigma `0.1` (`p=0.3`). Geometric transformations are applied jointly to images and masks.

## Figures and tables

All figure panels are saved as **SVG with editable text and white backgrounds**. Numerical outputs are exported as CSV, with an additional Excel workbook for statistical tables.

| Output location | Main contents |
|---|---|
| `preprocessed/` | Nuclear patches and masks, sample/split manifests, QC tables, size and count plots |
| `features/` | Original and augmented feature vectors, sample identities and sampling plans |
| `cv/` | Training, validation and test matrices for each fold and condition |
| `classical/`, `resnet/` | Fitted models/checkpoints, validation selections, predictions, metrics, confusion matrices and ROC curves |
| `statistics/` | Three-approach comparison plots with statistical brackets, condition comparisons, training histories, pooled evaluation panels and statistical tables |
| `representations/` | PCA/UMAP panels, Figure 5 heatmaps, coordinates, loadings and correlation tables |

## Running jobs in the background

On a server accessed through SSH/PuTTY, `nohup` allows a job to continue after the terminal is closed. Activate `nuclei_classification` and define the path variables first. For example, launch step 2 with:

```bash
nohup python -u scripts/2_embedding_extraction.py \
  --data "$RUN/preprocessed" --output "$RUN/features" \
  --families dino handcrafted --levels 100 200 500 1000 2000 4000 \
  --dino-repo "$DINO_REPO" --dino-config "$DINO_CONFIG" \
  --dino-weights "$DINO_WEIGHTS" \
  --device cuda --batch-size 16 --workers 8 \
  > "$RUN/2_embedding_extraction.log" 2>&1 < /dev/null &

echo $! > "$RUN/2_embedding_extraction.pid"
```

Monitor progress and check the process:

```bash
tail -f "$RUN/2_embedding_extraction.log"
# Press Ctrl+C to stop viewing the log; the background job continues.
ps -p "$(cat "$RUN/2_embedding_extraction.pid")" -o pid,etime,stat,cmd
nvidia-smi
```

Confirm that the log contains no startup error before disconnecting. Wait for a step to finish successfully before starting a dependent step. On managed clusters, follow the local scheduler requirements for GPU and CPU jobs.

## Reproducibility

Split identities, augmentation seeds, file hashes and validation selections are saved with the run. Keep these together with checkpoints, predictions and the software environment:

```bash
python -m pip freeze > "$RUN/environment.txt"
git -C "$DINO_REPO" rev-parse HEAD > "$RUN/3dino_revision.txt"
```

Use a new `RUN` directory for a new experiment. Feature extraction, CNN selection and analysis retain their existing new/empty-output requirements. Classical selection requires a new/empty output directory for a fresh search, or `--resume` / `--resume-classical` for a compatible checkpointed search. Resume is limited to classical selection; it is not a general pipeline restart mechanism.

Classical search provenance includes input and source-code hashes, the parameter grids, software versions, seed and numerical thread settings. Keep `search_config.json`, `selection.json` and fitted model artifacts together. Temporary preprocessing caches are not required for evaluating saved models.

Run the included checks with:

```bash
python -m pytest -q tests/test_scientific_contracts.py
python tests/synthetic_workflow.py --output "$RUN/synthetic_smoke" --cnn
```

The smoke workflow uses reduced synthetic data and training budgets. Its representation-plot vectors are synthetic and do not validate external 3DINO inference. See [Verification](docs/VERIFICATION.md) for tested components and execution limits.

## Repository structure

| Location | Purpose |
|---|---|
| `scripts/1_preprocessing.py` | Patch extraction, QC and split generation |
| `scripts/2_embedding_extraction.py` | 3DINO and handcrafted feature extraction |
| `scripts/3_cross_validation_data.py` | CV dataset assembly |
| `scripts/4_run_models.py` | Feature-based model selection and evaluation |
| `scripts/parallel_selection.py` | Parallel classical searches, checkpoints, resume checks and deterministic selection |
| `scripts/classical_models.py` | Estimators, training-only preprocessing and identity-isolated SVM calibration |
| `scripts/5_run_cNN.py` | Direct 3D model selection and evaluation |
| `scripts/dataset_helper.py` | Patch loading, spatial preparation and augmentation |
| Other modules in `scripts/` | Feature definitions, model architectures, grids, metrics and shared utilities |
| `scripts/6_statistical_analysis.py` | Model comparisons, statistical tables and evaluation panels |
| `scripts/7_representation_analysis.py` | PCA, UMAP and feature-correlation analysis |
| `tests/` | Scientific contract tests and synthetic workflow |
| `docs/` | Detailed feature definitions and verification information |
| `requirements.txt` | Python dependencies |
| `run_pipeline.sh` | Ordered workflow stages, classical parallelism controls and per-stage logs |
| `scripts/check_environment.py` | Dependency imports, DINO configuration and optional GPU forward check |

## Data and references

The study uses 3D mouse liver microscopy data provided by Universidad de Concepción, Chile. Dataset reference: [Zenodo, DOI: 10.5281/zenodo.19502784](https://doi.org/10.5281/zenodo.19502784).

For the pretrained representation model, cite:

Xu, T., Hosseini, S., Anderson, C., Rinaldi, A., Krishnan, R. G., Martel, A. L., & Goubran, M. (2025). **A generalizable 3D framework and model for self-supervised learning in medical imaging.** *npj Digital Medicine*. [https://doi.org/10.1038/s41746-025-02035-w](https://doi.org/10.1038/s41746-025-02035-w).

[3DINO source code](https://github.com/AICONSlab/3DINO) · [Pretrained weights](https://huggingface.co/AICONSlab/3DINO-ViT)
