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

Evaluation uses **five-fold leave-one-animal-out cross-validation**. In each fold, one animal is reserved for testing. Original nuclei from the other four animals are split into training and validation, with **10% of the development nuclei reserved for validation**, stratified jointly by animal and class. The validation count is rounded up to the next integer.

These splits are saved during preprocessing and shared by every model and augmentation condition. **Only training nuclei are augmented.** Validation and test nuclei remain unaugmented, and augmented copies of a nucleus cannot cross subsets within a fold.

Validation macro-F1 determines model selection, with balanced accuracy and lower log-loss used to break ties. Validation also selects augmentation levels and training checkpoints. The selected fitted model is evaluated on the held-out animal without a train-plus-validation refit. Selection is performed independently in each fold, so the selected configuration can differ between folds.

The primary comparison is between the three validation-selected approaches, using their held-out animal scores. Test scores are not used to select classifiers, augmentation counts or checkpoints. Exploratory PCA/UMAP and correlation analyses are kept separate from predictive model fitting.

## Installation

### 1. Activate the Python environment

The dependency versions in `requirements.txt` target **Python 3.12**. Run installation commands from the repository root, where `requirements.txt` is located.

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
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

The PyTorch command uses CUDA 12.4 wheels for the GPU server configuration. Other systems require a PyTorch build compatible with their hardware and drivers.

### 3. Install 3DINO and obtain the pretrained weights

**3DINO (also referred to here as DINO3D) must be installed separately before embedding extraction.** Its source code and pretrained weights are not included in this repository.

- Follow the installation instructions in the [official 3DINO repository](https://github.com/AICONSlab/3DINO).
- Make its required dependencies available inside `nuclei_classification`.
- Obtain the [pretrained 3DINO-ViT weights](https://huggingface.co/AICONSlab/3DINO-ViT) and their matching configuration.
- Set `DINO_REPO`, `DINO_CONFIG` and `DINO_WEIGHTS` as described below.

The loader uses `dinov2.configs.load_and_merge_config_3d` and `dinov2.eval.setup.build_model_for_eval`. Use a compatible 3DINO checkout and matching weights/configuration; the expected output is one 1,024-dimensional vector per patch. Record the checkout revision used for each experiment.

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
RUN="$PROJECT/results/publication_v2"
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

`PROJECT/code` should contain this repository's `notebooks/`, `statistic_notebooks/` and `requirements.txt`.

## Running the pipeline

Run the numbered steps in order. Complete **both model-selection stages before starting final evaluation**. Each script provides `--help` for its full command-line options.

### 1. Extract nuclear patches and define the splits

```bash
python notebooks/1_preprocessing.py \
  --images "$DATA/image" --instances "$DATA/labels" --classes "$DATA/class" \
  --output "$RUN/preprocessed" \
  --seed 42 --min-voxels 2500 --border-margin 3
```

This step checks the input volumes, excludes undersized and border-touching nuclei, normalizes DAPI intensities, and saves each nuclear patch with its original instance mask. It also generates size/count plots, QC tables and the shared train/validation/test split manifest.

**Output:** `$RUN/preprocessed`

### 2. Extract 3DINO embeddings and handcrafted features

```bash
python notebooks/2_embedding_extraction.py \
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
python notebooks/3_cross_validation_data.py \
  --data "$RUN/preprocessed" --features "$RUN/features" \
  --output "$RUN/cv"
```

This step assembles all five folds for both feature families, including the **original-only baseline** and every requested augmentation level. It uses the splits saved in step 1 and checks sample identities, labels and feature schemas.

**Output:** `$RUN/cv`

### 4. Train and select feature-based classifiers

```bash
python notebooks/4_run_models.py --phase select \
  --cv "$RUN/cv" --output "$RUN/classical"
```

Logistic Regression, Random Forest, SVM and MLP are fitted for each feature family, fold and augmentation condition. Feature scaling and predictive PCA are fitted on training data only. MLP early stopping uses the shared validation set. SVM probability calibration uses internal training folds grouped by original nucleus identity.

Validation scores select the fitted models, including the classifier and augmentation level for each representation family. Search results and the selected model identities are saved before test evaluation.

**Output:** `$RUN/classical`

### 5. Train and select the direct 3D CNN

```bash
python notebooks/5_run_cNN.py --phase select \
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
python notebooks/4_run_models.py --phase evaluate \
  --cv "$RUN/cv" --output "$RUN/classical"

python notebooks/5_run_cNN.py --phase evaluate \
  --data "$RUN/preprocessed" --output "$RUN/resnet" \
  --device cuda --workers 8
```

Evaluation loads the saved models/checkpoints and exports predictions, metrics, confusion matrices and ROC curves. The primary comparison uses the configuration already selected by validation in each fold. Results for all fixed classifier/augmentation conditions are also exported for descriptive comparisons.

### 6. Compare models and generate statistical summaries

```bash
python statistic_notebooks/6_statistical_analysis.py \
  --results "$RUN/classical" "$RUN/resnet" \
  --output "$RUN/statistics" --all-pairwise
```

The main plots compare the validation-selected **3DINO**, **handcrafted-feature** and **CNN** approaches on the same five held-out animals. They show individual animal scores and pairwise brackets labelled with Holm-adjusted Wilcoxon p-values.

Macro-F1 is the primary metric; weighted-F1 and balanced accuracy are complementary. Tables include paired effect sizes, model ranks and a within-animal permutation test of the Friedman rank statistic. Holm correction covers the tested pairs across all three comparison metrics within each exported comparison family.

Animal-level means give each animal equal weight. Pooled confusion matrices and ROC curves summarize individual nuclear predictions and therefore weight animals by their numbers of nuclei. With only five animals and overlapping CV training sets, inferential statistics should be interpreted as exploratory.

**Output:** `$RUN/statistics`

### 7. Visualize representations and feature correlations

```bash
python statistic_notebooks/7_representation_analysis.py \
  --features "$RUN/features" --output "$RUN/representations" \
  --seed 42 --neighbors 50 --min-dist 0.5
```

This analysis uses **original, unaugmented nuclei**, matched by identity across the two feature families. It generates:

- **PCA and UMAP panels:** class- and animal-labelled representation plots, PCA component pairs, scree plots, coordinates and loadings.
- **Figure 5A:** correlations between the first 31 principal components of standardized 3DINO embeddings and the 31 handcrafted features. PC labels include their explained variance.
- **Figure 5B:** correlations between the 31 highest-variance individual 3DINO embedding dimensions and the same handcrafted features. Dimensions are ranked by variance before standardization.
- **Supplementary outputs:** Spearman heatmaps and correlation tables stratified by animal and class.

These analyses describe representation structure and do not feed into model selection. Constant-feature correlations are displayed as undefined rather than assigned a numerical association.

**Output:** `$RUN/representations`

## Preprocessing and features

DAPI volumes are normalized independently using the **30th and 99.999th percentiles** and clipped to `[0, 1]`. Nuclear class labels are assigned by the majority nonzero class within each instance mask. Ties and exclusions are recorded in QC tables.

Each extracted patch is min–max normalized, center-padded to **70³ voxels**, or to a larger enclosing cube when necessary, and resized to **112³ voxels**. Images use `order=0` interpolation with antialiasing; masks use matching nearest-neighbor resizing without antialiasing. The transformed original instance mask defines the nucleus throughout feature extraction. DINO/CNN inputs are mapped to `[-1, 1]`; handcrafted intensity features use `[0, 1]`.

The 31 handcrafted features describe morphology, intensity, full 3D masked texture and finite-scale complexity. Mean radius is measured from the nuclear centroid to the surface, and shape index is estimated from local principal curvatures. Definitions and units are provided in [Feature definitions](docs/FEATURES.md).

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
nohup python -u notebooks/2_embedding_extraction.py \
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

Feature extraction, model selection and analysis require new or empty output folders. Use a new `RUN` directory for a new experiment; automatic resumption of partial runs is not implemented.

Run the included checks with:

```bash
python -m pytest -q tests/test_scientific_contracts.py
python tests/synthetic_workflow.py --output "$RUN/synthetic_smoke" --cnn
```

The smoke workflow uses reduced synthetic data and training budgets. Its representation-plot vectors are synthetic and do not validate external 3DINO inference. See [Verification](docs/VERIFICATION.md) for tested components and execution limits.

## Repository structure

| Location | Purpose |
|---|---|
| `notebooks/1_preprocessing.py` | Patch extraction, QC and split generation |
| `notebooks/2_embedding_extraction.py` | 3DINO and handcrafted feature extraction |
| `notebooks/3_cross_validation_data.py` | CV dataset assembly |
| `notebooks/4_run_models.py` | Feature-based model selection and evaluation |
| `notebooks/5_run_cNN.py` | Direct 3D model selection and evaluation |
| `notebooks/dataset_helper.py` | Patch loading, spatial preparation and augmentation |
| Other modules in `notebooks/` | Feature definitions, model architectures, grids, metrics and shared utilities |
| `statistic_notebooks/6_statistical_analysis.py` | Model comparisons, statistical tables and evaluation panels |
| `statistic_notebooks/7_representation_analysis.py` | PCA, UMAP and feature-correlation analysis |
| `tests/` | Scientific contract tests and synthetic workflow |
| `docs/` | Detailed feature definitions and verification information |
| `requirements.txt` | Python dependencies |

## Data and references

The study uses 3D mouse liver microscopy data provided by Universidad de Concepción, Chile. Dataset reference: [Zenodo, DOI: 10.5281/zenodo.19502784](https://doi.org/10.5281/zenodo.19502784).

For the pretrained representation model, cite:

Xu, T., Hosseini, S., Anderson, C., Rinaldi, A., Krishnan, R. G., Martel, A. L., & Goubran, M. (2025). **A generalizable 3D framework and model for self-supervised learning in medical imaging.** *npj Digital Medicine*. [https://doi.org/10.1038/s41746-025-02035-w](https://doi.org/10.1038/s41746-025-02035-w).

[3DINO source code](https://github.com/AICONSlab/3DINO) · [Pretrained weights](https://huggingface.co/AICONSlab/3DINO-ViT)
