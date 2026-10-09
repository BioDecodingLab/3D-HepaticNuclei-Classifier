# Tutorial: train and apply a frozen-3DINO nuclear morphotype classifier

This folder provides a practical, configuration-driven workflow for **training or applying the frozen-3DINO + classical-classifier approach** described in the accompanying study. It is deliberately separate from the publication pipeline: the paper uses five-fold leave-one-animal-out cross-validation, whereas this tutorial uses an explicit **training set** and an independent **test set**, with an optional third dataset for prediction on new segmented images.

The entire tutorial is executed with one command:

```bash
python Tutorial/run_tutorial.py --config Tutorial/config.toml
```

All paths, model choices and processing settings are defined in `config.toml`.

## What the tutorial does

When `run.train_model = true`, the workflow:

1. reads labeled training microscopy volumes, nuclear instance labels and reference class maps;
2. applies the same core QC and patch-standardization principles used in the paper;
3. reserves a validation subset **from the training data only**;
4. extracts frozen 3DINO representations from original and, when requested, augmented training nuclei;
5. fits the selected classical classifier (`logreg`, `rf`, `svm`, or `mlp`) using the same classifier-specific parameter grid as the publication code;
6. selects hyperparameters using validation macro-F1, with balanced accuracy and lower log loss used as tie-breakers;
7. saves the fitted classifier together with provenance and preprocessing metadata;
8. optionally evaluates the locked model on an independent labeled test directory; and
9. optionally predicts nuclear morphotypes for new segmented images.

When `run.train_model = false`, training is skipped. The workflow loads the model specified by `run.model_path`, verifies that the configured 3DINO weights/configuration match the saved model metadata, and then evaluates and/or predicts using the configured test and prediction datasets.

## Installation

Use the same `nuclei_classification` environment described in the repository root `README.md`. The tutorial intentionally reuses the repository's tested preprocessing helpers, classifier definitions, parameter grids and evaluation code. Therefore, keep the `Tutorial/` folder inside the repository root.

3DINO itself and its pretrained weights are external dependencies and are not bundled here. Set the following paths in `config.toml`:

```toml
[dino]
repo = "../3DINO"
config = "../3DINO/dinov2/configs/train/vit3d_highres"
weights = "./3dino_vit_weights.pth"
```

The upstream 3DINO code and weights remain subject to their own license and usage conditions.

## Input-data organization

The tutorial uses the same three logical inputs as the publication pipeline: an intensity image, a nuclear instance-label volume, and, for labeled datasets, a reference class map. Instead of cross-validation folds, the user supplies independent training and test directories.

A typical layout is:

```text
example_data/
├── train/
│   ├── image/
│   │   ├── train_01.tif
│   │   └── train_02.tif
│   ├── labels/
│   │   ├── train_01.tif
│   │   └── train_02.tif
│   └── class/
│       ├── train_01.tif
│       └── train_02.tif
├── test/
│   ├── image/
│   │   └── test_01.tif
│   ├── labels/
│   │   └── test_01.tif
│   └── class/
│       └── test_01.tif
└── predict/
    ├── image/
    │   └── new_01.tif
    └── labels/
        └── new_01.tif
```

The filename stems must match within a dataset. For example, `image/train_01.tif`, `labels/train_01.tif`, and `class/train_01.tif` describe the same volume.

### `image/`

Contains the microscopy intensity volume. Supported TIFF inputs are:

- **one channel:** a 3D array `(Z, Y, X)`;
- **two channels:** a 4D array with two channels;
- **three channels:** a 4D array with three channels.

For 4D TIFFs, `data.channel_axis = "auto"` attempts to use TIFF axis metadata or an unambiguous channel-sized dimension. If the axis cannot be inferred safely, set an explicit integer, for example `0` for `CZYX` or `-1` for `ZYXC`.

Every spatial channel must align exactly with the corresponding instance-label and class-label volumes.

### `labels/`

Contains a **3D nuclear instance-segmentation volume**. Background is `0`; each nucleus must have a unique positive integer ID. The classifier operates on individual segmented nuclei, so instance labels are required for training, testing and prediction. The tutorial does **not** run nuclear segmentation automatically.

### `class/`

Required for training and labeled test datasets. Contains the reference class map with:

| Code | Default morphotype |
|---:|---|
| 0 | background / unlabeled |
| 1 | Hepatocyte |
| 2 | Stellate cell |
| 3 | Kupffer cell |
| 4 | Endothelial cell |
| 5 | Other cell |

The class assigned to each nuclear instance is the majority nonzero class overlapping that instance, matching the publication preprocessing logic. Prediction datasets do not require a `class/` folder.

## One-, two- and three-channel inputs

The submitted paper evaluates **one-channel DAPI input only**. Multi-channel support in this tutorial is an extension for reuse on other datasets and should not be interpreted as a result validated by the paper.

The tutorial constructs the 3DINO input streams as follows:

- **1 raw channel:** use the single channel, as in the paper;
- **2 raw channels:** create three streams internally: channel 1, channel 2, and a normalized `channel 1 + channel 2` sum;
- **3 raw channels:** use the three supplied channels in their given order.

Each raw channel is percentile-normalized at the volume level, nuclear patches are extracted using the instance mask, and each patch is standardized to `112 × 112 × 112` voxels before frozen 3DINO inference.

### Important 3DINO channel compatibility

The official pretrained 3DINO-ViT architecture used in the paper is configured as a **one-channel volumetric model**. Therefore, a three-channel tensor cannot simply be passed into that exact checkpoint without changing its input layer.

To keep the pretrained backbone unmodified, the tutorial provides three strategies through `dino.channel_strategy`:

- `"auto"` (**recommended**): if the loaded backbone natively accepts the prepared number of streams, they are passed directly. If the backbone accepts one channel, each stream is processed independently by the same frozen 3DINO backbone and the resulting embeddings are concatenated.
- `"direct"`: all prepared streams are passed as one multi-channel tensor. This requires a 3DINO checkpoint/configuration whose patch embedding expects exactly that number of channels.
- `"per_channel_concat"`: each prepared channel is processed independently by a one-channel 3DINO backbone; the embeddings are concatenated before classical classification.

For the official one-channel 3DINO-ViT weights, `"auto"` therefore resolves to direct inference for one-channel data and per-channel concatenation for two-/three-channel data. This avoids silently modifying pretrained weights.

A model trained with one channel should be applied to one-channel prediction data; a model trained with two channels should be applied to two-channel data; and likewise for three channels. The tutorial records and enforces this in the saved model metadata.

## Training and test split

This tutorial does **not** perform leave-one-animal-out cross-validation. `data.training_dir` and `data.test_dir` are treated as separate datasets.

Within `training_dir`, a validation subset is created before augmentation. By default, the split is stratified jointly by training image and class:

```toml
[training]
validation_fraction = 0.10
stratify_validation_by_image_and_class = true
```

Only the remaining fit nuclei are eligible for augmentation. The independent test directory is never used for model or hyperparameter selection.

For biologically meaningful performance estimates, the test images should represent independent biological samples and should not be alternate crops or technical repeats of the training images.

## Augmentation level

Set:

```toml
augmentation_level = 4000
```

The interpretation matches the feature-based publication pipeline: the value is the **total number of sampled augmented patches per class per training image**. Sampling is with replacement from eligible training nuclei. An augmentation level of `0` uses only the original fit nuclei.

The levels evaluated in the paper were:

```text
0, 100, 200, 500, 1000, 2000, 4000
```

The tutorial accepts any non-negative integer, but using the publication levels facilitates comparison with the reported workflow.

Augmentation uses the same family of transformations as the paper: axis permutations, independent flips, intensity scaling/shifting, Gaussian blur and Gaussian noise. Spatial transforms are shared across all channels and the nuclear mask.

## Selecting the classical classifier

Choose one downstream classifier in `config.toml`:

```toml
classifier = "svm"
```

Supported values are:

- `"logreg"` — logistic regression;
- `"rf"` — random forest;
- `"svm"` — radial-basis-function support vector machine;
- `"mlp"` — multilayer perceptron.

With `hyperparameter_search = true`, the tutorial evaluates the same classifier-specific parameter grid used by the publication pipeline and selects the fitted candidate using the validation set. The independent test dataset is evaluated only after this selection is complete.

## Train a new model

Set:

```toml
[run]
train_model = true
```

and provide `data.training_dir`. `run.model_path` is ignored in training mode.

Run:

```bash
python Tutorial/run_tutorial.py --config Tutorial/config.toml
```

The trained model is written to:

```text
<output_dir>/trained_model/model.joblib
<output_dir>/trained_model/model_metadata.json
```

Keep these two files together. The metadata records the class names, preprocessing settings, raw channel count, resolved multi-channel strategy, embedding dimensionality, classifier settings, augmentation level, 3DINO configuration and weight checksums, and the 3DINO source revision when available.

## Use an existing trained model

Set:

```toml
[run]
train_model = false
model_path = "/path/to/trained_model"
```

`model_path` may point either to the directory containing `model.joblib` and `model_metadata.json` or directly to `model.joblib`.

The tutorial verifies that the configured 3DINO weights and configuration match those recorded when the classifier was trained. This prevents accidental inference with a different frozen representation model.

## Predict new images

Set:

```toml
[data]
prediction_dir = "/path/to/new_data"
```

The prediction directory requires only `image/` and `labels/`. Outputs include:

```text
<output_dir>/predictions/predictions.csv
<output_dir>/predictions/predicted_class_volumes/
<output_dir>/predictions/embeddings.npz          # when save_embeddings = true
```

`predictions.csv` contains the image ID, nuclear instance ID, predicted class and all five class probabilities. `predicted_class_volumes/` contains a TIFF class map in the original segmentation geometry, where each retained nuclear instance is assigned its predicted class code and all other voxels are `0`.

## Independent test outputs

When a labeled `test_dir` is provided, the tutorial saves:

- aggregate metrics;
- per-class precision, recall and F1;
- confusion matrices in CSV and SVG format;
- one-vs-rest ROC outputs;
- per-nucleus predictions and probabilities;
- predicted class-label TIFF volumes; and
- frozen embeddings when `save_embeddings = true`.

These are written under:

```text
<output_dir>/test_evaluation/
```

## Main configuration fields

| Section | Field | Purpose |
|---|---|---|
| `[run]` | `train_model` | `true` to train; `false` to load a saved classifier |
| `[run]` | `model_path` | Existing tutorial model when `train_model=false` |
| `[run]` | `output_dir` | New output directory |
| `[data]` | `training_dir` | Labeled training dataset |
| `[data]` | `test_dir` | Optional independent labeled test dataset |
| `[data]` | `prediction_dir` | Optional unlabeled dataset for prediction |
| `[data]` | `channel_axis` | `"auto"` or explicit axis for 4D TIFFs |
| `[training]` | `classifier` | `logreg`, `rf`, `svm`, or `mlp` |
| `[training]` | `augmentation_level` | Augmented samples per class per training image |
| `[training]` | `validation_fraction` | Internal validation fraction within training data |
| `[dino]` | `repo` | 3DINO source checkout |
| `[dino]` | `config` | 3DINO model configuration |
| `[dino]` | `weights` | Pretrained 3DINO weights |
| `[dino]` | `channel_strategy` | `auto`, `direct`, or `per_channel_concat` |

## Reproducibility and safeguards

The workflow is designed to fail loudly rather than silently reinterpret inconsistent inputs. It checks matching TIFF stems and spatial shapes, valid class codes, channel counts, nonempty QC-filtered datasets, validation coverage of all five classes, 3DINO input-channel compatibility, embedding dimensions, and model/config/weight consistency when loading a saved model.

The output directory also contains:

```text
run.log
run_metadata.json
resolved_model_metadata.json
train_validation_split.csv             # training mode
augmentation_plan.csv                   # when augmentation_level > 0
classifier_search/validation_search.csv # training mode
cache/                                  # extracted native-size nuclear patches
```

Use a new output directory for each run. By default the tutorial refuses to write into a nonempty output directory. Set `run.overwrite_output = true` only when intentional.

## Relationship to the paper pipeline

This tutorial is intended for **application and reuse**, not for reproducing the manuscript's model-comparison statistics. The manuscript pipeline remains the authoritative implementation for the five-fold leave-one-animal-out benchmark, fold-specific validation selection, comparison against handcrafted features and ResNet3D-18, and the reported statistical analyses.

The tutorial preserves the key application logic of the frozen-3DINO approach while replacing cross-validation with explicit training/test datasets and adding a deployment-oriented prediction mode.
