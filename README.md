# 3D hepatic nuclear morphotype classification

A corrected, auditable implementation of the supplied hepatic nuclei pipeline.
The methodology retains frozen 1024-dimensional 3DINO embeddings, 31 handcrafted
DAPI-derived descriptors, Logistic Regression / Random Forest / SVM / MLP, and
the original ResNet3D-18-style direct classifier. The optional DINO classification
head architecture remains available. No biological results are supplied or claimed
by this release. All experiments must be rerun.

## Experimental contract

- Five images represent five distinct animals. Source voxel spacing is isotropic:
  **0.3 × 0.3 × 0.3 µm**. Class codes: 1 Hepatocyte, 2 Stellate, 3 Kupffer,
  4 Endothelial, 5 Other. Input `--images` must contain DAPI-only 3D arrays.
- Outer evaluation is leave-one-animal-out. Within each fold, exactly
  `ceil(0.10 × number of development nuclei)` originals form validation.
  Splitting is stratified jointly by animal and class. If strata cannot be
  represented, preprocessing stops; it never silently changes the fraction.
- Step 1 saves one immutable split manifest. Every representation, augmentation
  condition, classifier and direct model uses the same original identities in
  that fold. Test animals are absent from both training and validation.
- Training is fitted only on training data. Validation selects hyperparameters,
  MLP epochs, direct-model checkpoints, classifier family members, and augmentation
  levels. **There is no train + validation refit.**
- Selection and test evaluation are distinct command-line phases. Complete all
  planned model searches before running either evaluation phase. Save and freeze
  the protocol before inspecting test outputs. Test metrics must not drive reruns,
  thresholds, architecture choices or a new global winner.
- Selecting a single configuration by averaging validation results across outer
  folds is not used: an outer test animal participates in development of other
  folds. The primary estimand is the performance of the fold-specific selection
  procedure. A different configuration may win in each fold.
- A test animal can be a training animal in another outer fold; this is the
  intended CV design. Within any one fold it is completely held out.
- PCA/UMAP and Figure 5 are post hoc original-only exploration. They produce no
  predictive transforms and are not inputs to training.

## Numbered workflow

```text
1_preprocessing.py
  original DAPI + instances + class maps
  -> intensity/mask patches, QC, inspection SVGs, sample table, frozen splits
2_embedding_extraction.py
  -> original caches and training-only augmented features for all six levels
3_cross_validation_data.py
  -> CV matrices, including original-only baseline
4_run_models.py --phase select
5_run_cNN.py --phase select
  -> locked validation selections and fitted models/checkpoints
4_run_models.py --phase evaluate
5_run_cNN.py --phase evaluate
  -> held-out predictions, metrics, confusion matrices and ROC SVGs
6_statistical_analysis.py
  -> animal-level comparisons, corrected statistical tables and SVG panels
7_representation_analysis.py
  -> PCA, UMAP, Figure 5A/B, correlation tables and supplementary summaries
```

The three old preprocessing/inspection scripts are merged into step 1. Step 5
starts from step 1 outputs, independently of the feature-classifier branch.
The unsuccessful `embedding_opt_*` experiment is deliberately omitted.

## Installation

Use Python 3.12 in an isolated environment. The CPU numerical checks for this
release were performed with the installed versions recorded in
`docs/VERIFICATION.md`; the external 3DINO loader and real GPU workload must also
be checked on the target server.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Existing server configuration: PyTorch 2.6.0, CUDA 12.4 wheels.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only verification, install a compatible CPU PyTorch wheel instead.
The external 3DINO repository is not redistributed. Install its own requirements
and resolve compatibility within the same environment before extraction. Record
`pip freeze`, its git revision, configuration and checkpoint checksum for each
real experiment. The code imports the same external functions as the original:
`dinov2.configs.load_and_merge_config_3d` and
`dinov2.eval.setup.build_model_for_eval`.

The README does not claim that an arbitrary new external 3DINO revision is
compatible with these entry points. Use the checkout that supplied your original
weights/configuration. A strict output check requires `(batch, 1024)` embeddings.

## Paths and runnable commands

Run from this project directory. These variables preserve the original server
locations while placing corrected outputs in a new tree to avoid overwriting
previous experiments. No site-specific absolute paths are embedded in Python.

```bash
PROJECT=/medicina/hmorales/projects/Nuclei3DClassification
DATA="$PROJECT/data"
RUN="$PROJECT/results/publication_v2"
DINO_REPO="$PROJECT/code/3DINO"
DINO_CONFIG="$DINO_REPO/dinov2/configs/train/vit3d_highres"
DINO_WEIGHTS="$DATA/3dino_vit_weights.pth"
mkdir -p "$RUN"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Original source folders were `$DATA/image`, `$DATA/labels` and `$DATA/class`.
Within them, the matching TIFF stems identify animals (e.g. `1.tif` in all three).
Both `.tif` and `.tiff` are supported. Image/instance/class arrays must have the
same shape; instance labels must be nonnegative integers and class maps 0..5.

### 1. Extract and inspect originals; freeze validation

```bash
python notebooks/1_preprocessing.py \
  --images "$DATA/image" --instances "$DATA/labels" --classes "$DATA/class" \
  --output "$RUN/preprocessed" --seed 42 --min-voxels 2500 --border-margin 3
```

Border-touching objects are excluded at the original three-voxel margin. The
majority nonzero class inside the exact instance mask defines the class; ties
retain the original smallest-code rule and are flagged in `object_qc.csv`.
All exact mask voxels, including dark voxels and original holes, are preserved.
No hole filling, Otsu or intensity-derived mask is used for feature geometry.

Intensity volumes retain original percentile normalization (30, 99.999) and
clipping to [0,1]. `samples.csv` records original instance IDs, class, animal,
relative paths, hashes, dimensions, bounding-box and mask voxel counts, and original
volume in µm³. `split_assignments.csv` makes the saved splits easy to inspect.
Plots include depth/height/width, bounding-box and mask volumes, class distributions
and animal-class counts. Mask and intensity trees are separate so masks cannot be
accidentally rediscovered as intensity inputs.

A nonempty output directory is rejected. Use a new run directory if parameters
change; this prevents stale patches, mixed schemas and silently reused splits.

### 2. Extract both representations and every augmentation level

```bash
python notebooks/2_embedding_extraction.py \
  --data "$RUN/preprocessed" --output "$RUN/features" \
  --families dino handcrafted --levels 100 200 500 1000 2000 4000 \
  --dino-repo "$DINO_REPO" --dino-config "$DINO_CONFIG" \
  --dino-weights "$DINO_WEIGHTS" --device cuda --batch-size 16 --workers 4
```

The target is **augmented rows per class per training animal**, not new unique
nuclei and not originals plus extra rows. Four training animals imply `4 × 5 × T`
training rows, e.g. 80,000 rows at T=4000. Sampling is with replacement from the
original training identities. Some originals may not be sampled at small T.
The original-only condition is generated separately in step 3.

Plans are nested: the first 100 draws in each animal/class occur in the 200-level
condition, and so on. The largest requested level is extracted once per fold and
family, then subsetted. DINO and handcrafted extraction use identical sampled
identities, transformations and augmentation seeds. Plans are saved as CSV.
Validation/test are never sampled or augmented. Original caches include every
original once and can be reused across outer folds because extraction is fixed
and contains no dataset-fitted predictive transformation.

Frozen DINO uses eval/inference mode and no gradient updates. The feature-family
output directories are separate. The inference batch size defaults to 16 for
memory control; this is not the direct-model training batch size. Handcrafted
workers compute full 3D geometry, texture and complexity and may take substantial
time; benchmark a small run on the target machine first. Feature failures abort
with the nucleus identity rather than silently imputing or dropping test samples.

**Spatial standardization is retained:** per-patch min–max normalization; pad
small patches to 70³, or larger patches to a cube of their largest dimension;
resize to 112³ with the original `order=0`, `preserve_range=True`,
`anti_aliasing=True` intensity call. Masks receive corresponding nearest-neighbor
resizing without antialiasing. Values outside the transformed original mask are
zero. DINO/CNN tensors are mapped to [-1,1]; handcrafted intensity descriptors use
[0,1] explicitly. This feature-range clarification is an intentional change.

**Units:** the 31 features are measured in standardized voxel coordinates.
They are not original physical nuclear dimensions. A source-dependent resize
factor applies when the padding cube changes; do not assign 0.3 µm to output
voxels indiscriminately. Original physical volume is exported separately.
`docs/FEATURES.md` defines every descriptor and its limitations.

Augmentation retains axis permutations, independent flips (p=0.3), intensity
scale 0.9–1.1 and shift −0.05–0.05 (joint probability 0.3), Gaussian blur
sigma 0.4–1.2 (p=0.3), and Gaussian noise sigma 0.1 (p=0.3). Permutations/flips
include reflections. Geometry transforms image and mask together. Photometry
never redefines the mask. The old README's ±20%/±0.1 claims were inaccurate.

### 3. Assemble every condition automatically

```bash
python notebooks/3_cross_validation_data.py \
  --data "$RUN/preprocessed" --features "$RUN/features" --output "$RUN/cv"
```

This produces level 0 (original-only) plus all requested levels. NPZ keys include
`X_train/y_train/ids_train`, `X_val/y_val/ids_val`, `X_test/y_test/ids_test`,
original training features/IDs for safe SVM calibration, feature names and animal
IDs. The assembler never makes a split. It verifies identities, labels, counts,
feature schemas and finite values before saving.

### 4. Select classical models using validation only

```bash
python notebooks/4_run_models.py --phase select \
  --cv "$RUN/cv" --output "$RUN/classical"
```

The supplied parameter grids are retained in `model_grids.py`. StandardScaler
and PCA are fitted on training only. PCA uses the full solver explicitly for
reproducibility. Macro-F1 (fixed five classes), balanced accuracy and then lower
log-loss define lexicographic selection; an exact tie retains the first candidate
in deterministic traversal order.

MLP architecture/grid are retained, but incremental training uses the explicit
clean validation set (maximum 500 epochs, patience 15). The built-in internal
random validation split is disabled. The best fitted weights are saved, without
refitting on validation.

SVM keeps its original classifier/kernel/grid. Built-in ungrouped probability
calibration is replaced by original-identity-grouped OOF calibration. Every
augmented relative of a held-out calibration nucleus is excluded from the
internal scaler/PCA/SVM fit. Calibration scores use one original view per sampled
training identity. Up to five stratified folds are used (fewer only when unique
training class counts require it; at least two). A fixed multinomial logistic
calibrator (C=1, max_iter=5000) learns from these OOF scores. The final SVM fits the
supplied training rows only. This is an explicit probability-method correction;
SVM class predictions remain its own `predict`, while probabilities support AUC
and log-loss and can have a different argmax. Validation/test never calibrate.

`selection.json` locks each model/level's best candidate and the overall best
candidate in each family/fold. It records model and CV-file hashes. Search tables
retain all validation scores. No test prediction is made during selection.

### 5. Select the direct ResNet checkpoint using the same validation

```bash
python notebooks/5_run_cNN.py --phase select \
  --data "$RUN/preprocessed" --output "$RUN/resnet" \
  --model resnet3d_18 --device cuda --batch-size 256 --workers 4 --data-parallel
```

The original 32-base-channel residual architecture and dropout/lr/weight-decay
grid are retained. AdamW, weighted cross-entropy, ReduceLROnPlateau and mixed
precision are retained. Maximum epochs 100; early-stopping patience 5; scheduler
patience 2. Class weights are computed only from original training counts.
Training loss aggregation uses the correct sum-of-class-weights denominator.

The example preserves the original training batch size of 256. If memory is
insufficient, reduce `--batch-size` explicitly and record that experimental
change. Workers and extraction batch size are resource controls; changing the
CNN training batch size may change optimization and results.

Only original training nuclei are used by this baseline, with epoch-dependent
seeded augmentation. This differs from the old helper's same-transform-every-epoch
behavior and is documented as an intentional improvement. Its training dataset
is not forced to match the six resampled classical conditions. Validation is 10%,
not the old 15%, and uses exactly the same IDs as step 4. The validation-selected
checkpoint is retained; there is no final training-loss-based retraining stage.

The optional original DINO+head architecture is invoked with `--model dino_cls`
and the three DINO arguments used in step 2. It is an additional experiment,
not required for the three primary families. Frozen-backbone candidates remain
in eval mode during head training. Fine-tuned candidates receive gradients.

### Final evaluation: only after all selection phases are complete

```bash
python notebooks/4_run_models.py --phase evaluate \
  --cv "$RUN/cv" --output "$RUN/classical"
python notebooks/5_run_cNN.py --phase evaluate \
  --data "$RUN/preprocessed" --output "$RUN/resnet" --device cuda --workers 4
```

Evaluation loads saved weights and checks their hashes. It never fits a model.
Classical evaluation exports every already-locked model/level condition to retain
augmentation comparison plots. These fixed-condition results are descriptive;
they are not used to choose the reported family procedure. The primary family
result follows the pre-recorded `selected` flag in every outer fold.

### 6. Statistical tables and publication panels

```bash
python statistic_notebooks/6_statistical_analysis.py \
  --results "$RUN/classical" "$RUN/resnet" --output "$RUN/statistics" --all-pairwise
```

Outputs include per-fold/all-condition/selected-family summaries, paired matrices,
raw and Holm-adjusted Wilcoxon p-values, paired d_z, an exact (or Monte Carlo)
within-animal permutation rank test, descriptive ranks, Excel tables, SVG boxplots,
training histories, and pooled selected-procedure confusion/ROC panels. Duplicate
rows and incomplete pairing fail rather than being averaged/dropped silently.
The selected-family boxplots include pairwise brackets labelled with the actual
Holm-adjusted p-values, including nonsignificant results. Fixed-condition plots
remain descriptive; their optional tests are exported in tables.
Holm correction covers all tested pairs across the three comparison metrics in
each exported comparison family. No data-dependent baseline is called the winner.

Equal-weight animal macro-F1 is primary. Pooled nuclear metrics differ because
large animals contribute more nuclei. Five-class macro metrics include an absent
class as zero, with explicit support. Balanced accuracy averages recalls of
represented true classes. A class-specific AUC is undefined if that animal lacks
positives or negatives; strict five-class macro AUC is then unavailable, and a
separately named present-class summary is provided. No arbitrary AUC is imputed.

With five animals, effect sizes and inferential results are exploratory. The
minimum exact two-sided Wilcoxon p-value for five nonzero pairs is 0.0625.
Overlapping CV training sets limit independence; using a permutation rank test
instead of asymptotic Friedman does not remove that limitation. Report animal
scores and uncertainty transparently; do not claim significance from uncorrected
post-selection tests.

### 7. Figure 2 and Figure 5 components

```bash
python statistic_notebooks/7_representation_analysis.py \
  --features "$RUN/features" --output "$RUN/representations" \
  --seed 42 --neighbors 50 --min-dist 0.5
```

Run after the predictive protocol is frozen. Both original feature families are
matched by stable nucleus identity. Every original appears once; augmented
relatives do not inflate the exploratory sample size. Standardized PCA component
pairs (PC1/2, PC1/3, PC2/3), scree plots, UMAP, coordinates and loadings are saved.

Figure 5A: Pearson correlations between the first 31 PCs of standardized original
DINO embeddings and the 31 descriptors. Row labels include explained variance.
Figure 5B: Pearson correlations between the 31 highest-variance raw embedding
dimensions and the same descriptors. Dimension variance is calculated **before**
standardization; indices are explicitly zero-based. Correlation color limits are
fixed to [-1,1]. Spearman panels and per-animal/per-class correlation CSVs are
supplementary. Constant-column correlations remain undefined (gray), not zero.
All choices here are exploratory and never flow into predictive fitting.

These correlations do not demonstrate causality or disentangled biological
encoding. Pooled associations can reflect class composition and animal effects;
inspect supplementary stratified correlations. No nucleus-level p-value claims
are made. All panels are SVG with editable text and opaque white backgrounds.

## Verification and execution limits

```bash
python -m pytest -q tests/test_scientific_contracts.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tests/synthetic_workflow.py --output /tmp/hepatic-smoke-new --cnn
```

The synthetic workflow writes a new directory and uses reduced sizes, targets and
training budgets. Its 1024D Figure 5 input is explicitly synthetic, **not** a
substitute for validating 3DINO inference. Smoke outputs are flagged and rejected
by statistics unless `--allow-smoke` is explicitly passed. Never report these
outputs as experimental findings.

Every script provides `--help`. Optional `--folds` / `--levels` support targeted
verification; final statistics require all five folds. Nonempty extraction,
selection and analysis output directories are rejected to prevent mixing runs.
This release intentionally prioritizes immutable complete runs over automatic
resume. Keep successful large intermediates backed up; a future resume feature
must verify full provenance rather than trust filenames.

See `CHANGELOG.md`, `docs/MIGRATION.md`, `docs/FEATURES.md`, and
`docs/VERIFICATION.md`. The original
archive contains no license; no new claim of licensing rights is made here.
