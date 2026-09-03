# 3D-HepaticNuclei-Classifier

A reproducible pipeline for the classification of hepatic 3D nuclear morphotypes from DAPI microscopy using self-supervised 3DINO embeddings, handcrafted nuclear features, and direct 3D deep learning.

## Overview

This repository contains the computational pipeline used to classify hepatic nuclear morphotypes from 3D DAPI microscopy images.

Three methodological approaches were evaluated:

- **3DINO embeddings:** frozen 1024-dimensional self-supervised representations followed by classical machine-learning classifiers.
- **Handcrafted nuclear features:** DAPI-derived morphometric, intensity, texture, and complexity descriptors followed by classical machine-learning classifiers.
- **Direct patch classification:** a ResNet3D-18 model trained directly on 3D DAPI nuclear patches.

The five hepatic nuclear morphotype classes were:

1. Hepatocyte
2. Stellate cell
3. Kupffer cell
4. Endothelial cell
5. Other cell

Auxiliary fluorescence channels were used to define ground-truth labels during dataset construction, whereas all computational classification experiments used only DAPI-derived nuclear information as model input.

---

## Experimental design

The dataset consisted of five 3D mouse liver images, each obtained from a different animal.

Model evaluation was performed using **leave-one-animal-out cross-validation**. Because each 3D image was acquired from a different mouse, this was equivalent to leave-one-image-out cross-validation and resulted in five held-out animal test folds.

For the 3DINO and handcrafted-feature approaches, class-balanced augmented training datasets containing:

- 100 samples per class
- 500 samples per class
- 1000 samples per class
- 2000 samples per class
- 4000 samples per class

were evaluated.

For these classical machine-learning pipelines, **10% of the non-held-out training pool was reserved for validation** and used for hyperparameter selection.

For the direct ResNet3D-18 baseline, **15% of the non-held-out data was reserved for validation**.

In all approaches, the held-out animal was kept separate from training, validation, augmentation, and hyperparameter selection and was used only for final test evaluation.

The classical classifiers evaluated were:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Multilayer Perceptron (MLP)

**Macro-F1** was used as the primary model-selection metric because it gives equal weight to all five nuclear morphotype classes and is therefore appropriate for the strongly imbalanced dataset.

Weighted-F1 and balanced accuracy were used as complementary evaluation metrics.

---

## Data preprocessing

### Nuclear patch extraction

Individual nuclei were extracted from 3D DAPI microscopy volumes using the corresponding nuclear instance-label and class-map volumes.

The preprocessing pipeline:

- verifies that image, instance-label, and class-map volumes have matching dimensions,
- removes nuclei touching image borders,
- excludes objects below the minimum voxel-size threshold,
- assigns each segmented nucleus to its corresponding reference class,
- masks voxels outside the segmented nucleus,
- and saves each nucleus as an individual 3D TIFF patch.

### Intensity normalization

Before patch extraction, DAPI intensity volumes were percentile-normalized using the **30th and 99.999th percentiles** and clipped to the range `[0, 1]`.

During model-input preparation, each nuclear patch was additionally normalized to `[0, 1]`.

After spatial preprocessing and augmentation, input intensities were mapped to `[-1, 1]` for the 3DINO pipeline.

### Spatial standardization

Single-nucleus patches have variable original dimensions.

To generate standardized model inputs, patches were:

1. center-padded to obtain a cubic representation while preserving the complete nucleus,
2. resized to a final input volume of **112 × 112 × 112 voxels**.

The same spatial preprocessing logic was used for the 3DINO, handcrafted-feature, and direct 3D image-classification workflows.

---

## Data augmentation

Class-balanced augmented training datasets were generated only from the non-held-out training animals.

The 3D augmentation pipeline included:

- random 3D axis permutations,
- random flips along the three spatial axes,
- random intensity scaling,
- random intensity shifting,
- 3D Gaussian blurring,
- additive Gaussian noise.

The implemented intensity perturbations use:

- intensity scaling of approximately ±20%,
- intensity shifting of ±0.1,
- Gaussian blur with sigma sampled between 0.4 and 1.2,
- Gaussian noise with sigma = 0.1.

Intensity perturbation, Gaussian blur, and Gaussian noise are applied stochastically.

Augmentation is applied only to training samples. Validation and held-out animal test samples remain non-augmented.

---

## 3DINO embedding extraction

Self-supervised 3D representations were extracted using the pretrained **3DINO** framework.

The 3DINO backbone was used strictly as a **frozen representation extractor**:

- pretrained model weights were loaded,
- the model was placed in evaluation mode,
- no gradient updates were applied to the backbone,
- and each DAPI-only nuclear patch was converted into a 1024-dimensional feature embedding.

Embeddings were saved together with:

- class labels,
- source patch paths,
- and image/animal identifiers where applicable.

These embeddings were subsequently used to train the classical machine-learning classifiers.

---

## Handcrafted nuclear features

As an interpretable baseline, handcrafted features were extracted from the same DAPI-only nuclear patches.

The feature set contains descriptors related to:

- nuclear volume,
- surface area,
- elongation,
- sphericity,
- equivalent diameter,
- bounding-box volume,
- extent,
- intensity statistics,
- texture,
- Haralick-like descriptors,
- fractal properties,
- lacunarity,
- surface properties,
- intensity-gradient-related measurements.

These features were extracted exclusively from DAPI-derived nuclear information, allowing direct comparison with the DAPI-only 3DINO representation.

---

## Cross-validation

The study used **five-fold leave-one-animal-out cross-validation**.

For each fold:

- one animal/image was held out exclusively for testing,
- the remaining four animals/images formed the model-development pool,
- model selection was performed without using the held-out test animal.

For the 3DINO and handcrafted-feature pipelines:

- 10% of the training pool was used for validation,
- hyperparameters were selected using validation macro-F1,
- the selected model was subsequently refitted using the available train + validation data,
- final performance was evaluated on the held-out animal.

For ResNet3D-18:

- the non-held-out animals were divided into training and validation subsets,
- 15% was reserved for validation,
- augmentation was applied only to the training subset,
- validation and held-out test nuclei remained non-augmented.

This animal-level evaluation design prevents nuclei from the held-out animal from contributing to model development.

---

## Classical machine-learning models

Four classical machine-learning classifiers were evaluated on both 3DINO embeddings and handcrafted features:

- Logistic Regression
- Random Forest
- Support Vector Machine
- Multilayer Perceptron

The machine-learning pipeline included:

1. feature standardization,
2. principal component analysis (PCA),
3. classification.

Hyperparameter optimization was performed independently within each cross-validation fold.

Model selection was based primarily on validation **macro-F1**, with balanced accuracy and log-loss used as secondary criteria when appropriate.

After hyperparameter selection, the selected configuration was evaluated on the corresponding held-out animal test set.

---

## Direct ResNet3D-18 baseline

A direct 3D deep-learning baseline was implemented to compare frozen representation learning with supervised classification directly from DAPI nuclear patches.

The ResNet3D-18-style architecture contains:

- 3D convolutional layers,
- batch normalization,
- ReLU activations,
- residual blocks,
- adaptive average pooling,
- dropout,
- a final fully connected classification layer.

The model was trained directly on the standardized 3D DAPI patches.

Each animal/image was used once as the held-out test fold.

Only the training subset received stochastic data augmentation.

---

## Statistical analysis

Model performance was evaluated across the five held-out animal test folds.

The primary endpoint was:

- **Macro-F1**

Complementary metrics included:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted-F1
- Macro precision
- Macro recall
- Balanced accuracy
- Macro ROC-AUC using one-vs-rest classification where applicable

For comparisons across augmentation levels and classical classifiers, the statistical workflow included:

- Friedman tests,
- average fold-wise model ranks,
- paired Wilcoxon signed-rank tests,
- Holm correction for multiple comparisons.

For the final comparison between the best models from the three methodological approaches, paired fold-level statistics were used.

Paired Cohen's d was calculated from fold-wise performance differences to quantify effect size.

Because only five held-out animals were available, inferential statistical results were interpreted cautiously, and statistical significance was considered together with observed fold-wise performance and effect sizes.

---

## PCA analysis

Principal component analysis was used to explore the organization of the learned and handcrafted DAPI-derived representation spaces.

Features were standardized before PCA.

PCA was used as an exploratory visualization tool to examine:

- class organization,
- overlap between hepatic nuclear morphotypes,
- animal/image-related patterns,
- and differences in representation structure.

PCA visualization was not used as quantitative evidence of classification performance.

Final predictive performance was assessed exclusively using held-out animal-level test folds.

---

## Results

The best-performing configuration from each methodological family was:

| Feature set / approach | Selected model | Accuracy | Weighted-F1 | Macro-F1 | Balanced Accuracy | AUC Macro OVR |
|---|---|---:|---:|---:|---:|---:|
| 3DINO embeddings | SVM + 2000 | 0.8333 | 0.8409 | 0.6132 | 0.6417 | 0.9272 |
| Direct patch classification | ResNet3D-18 | 0.8326 | 0.8228 | 0.5352 | 0.5360 | — |
| Handcrafted features | RF + 4000 | 0.7626 | 0.7753 | 0.5157 | 0.5521 | 0.9037 |

The **3DINO + SVM model trained using 2000 augmented samples per class** achieved the highest macro-F1 and balanced accuracy among the three selected methodological approaches.

Across the five held-out animal test folds, its performance was:

- **Macro-F1:** 0.613 ± 0.043
- **Weighted-F1:** 0.841 ± 0.037
- **Balanced accuracy:** 0.642 ± 0.051

Fold-wise performance differences and paired effect sizes favored the 3DINO-based approach.

However, pairwise Wilcoxon signed-rank comparisons did not remain statistically significant after Holm correction, consistent with the limited number of held-out animals.

The results therefore support a cautious interpretation: frozen self-supervised 3DINO representations provided stronger DAPI-only nuclear morphotype classification than the selected handcrafted-feature and direct ResNet3D-18 baselines, while larger independent datasets are required for stronger statistical confirmation.

---

## Running the project

### 1. Data preprocessing

**Files:**

- `notebooks/1_preprocessing.py`
- `notebooks/dataset_helper.py`

The preprocessing pipeline:

- loads 3D microscopy volumes,
- processes nuclear instance labels and class maps,
- removes border-touching or undersized nuclei,
- extracts individual nuclear patches,
- normalizes DAPI intensities,
- and saves class-organized TIFF patches.

---

### 1.1 Patch-size inspection

**Files:**

- `notebooks/1_1_get_patches_sizes.py`
- `notebooks/1_1_get_patches_sizes_all.py`

These scripts inspect the original 3D nuclear bounding-box dimensions and generate descriptive statistics and histograms used to evaluate the spatial characteristics of the extracted patches.

---

### 2. 3DINO embedding and handcrafted-feature extraction

**File:**

- `notebooks/2_embedding_extraction.py`

This script:

- loads preprocessed 3D DAPI nuclear patches,
- applies spatial standardization,
- performs training augmentation when configured,
- generates class-balanced augmented datasets,
- extracts frozen 3DINO embeddings or handcrafted nuclear features,
- and saves the resulting feature vectors and associated metadata.

The augmentation target can be configured according to the experimental condition being evaluated:

- 100
- 500
- 1000
- 2000
- 4000 samples per class.

---

### 3. Cross-validation split generation

**File:**

- `notebooks/3_cross_validation_data.py`

This script generates the five leave-one-animal-out cross-validation folds.

For each fold:

- one animal/image is used exclusively for testing,
- the remaining four animals/images form the model-development pool,
- 10% of the classical-model development pool is reserved for validation,
- the held-out animal test data are loaded from the non-augmented dataset.

---

### 4. Classical machine-learning models

**File:**

- `notebooks/4_run_models.py`

The following classifiers are evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine
- Multilayer Perceptron

The script performs:

- StandardScaler normalization,
- PCA,
- fold-specific hyperparameter search,
- validation-based model selection,
- final train + validation refitting,
- held-out test evaluation,
- per-class classification reports,
- confusion matrices,
- ROC curves and AUC where applicable.

---

### 5. Direct 3D deep-learning baseline

**File:**

- `notebooks/5_run_CNN.py`

This script implements direct classification of DAPI nuclear patches using ResNet3D-18.

For each held-out animal fold:

- the remaining animals form the development set,
- 15% is reserved for validation,
- augmentation is applied only to the training subset,
- model hyperparameters are selected using validation performance,
- final performance is evaluated on the non-augmented held-out animal.

---

### 6. Statistical analysis and model comparison

**File:**

- `statistic_notebooks/boxplot_stadistic.py`

This script performs the final statistical analysis used in the study, including:

- aggregation of held-out test-fold metrics,
- comparison of augmentation levels,
- comparison of classical classifiers,
- Friedman tests,
- average model ranks,
- paired Wilcoxon signed-rank tests,
- Holm correction for multiple comparisons,
- paired Cohen's d effect sizes,
- comparison of the selected 3DINO, handcrafted-feature, and ResNet3D-18 models,
- generation of publication-oriented performance plots.

---

### 7. PCA analysis

**Files:**

- `statistic_notebooks/PCA.ipynb`
- `statistic_notebooks/pca.py`

These analyses perform principal component analysis of DAPI-derived nuclear representations.

PCA is used to visualize representation structure across:

- hepatic nuclear morphotype classes,
- animals/images,
- and selected augmentation conditions.

---

## Dataset

The study uses 3D fluorescence microscopy volumes of mouse liver tissue provided by Universidad de Concepción, Chile.

The data supporting this study are publicly available on Zenodo:

https://doi.org/10.5281/zenodo.19502784

---

## 3DINO reference

The self-supervised embeddings were generated using the 3DINO framework:

Xu, T., Hosseini, S., Anderson, C., Rinaldi, A., Krishnan, R. G., Martel, A. L., & Goubran, M. (2025).  
*A generalizable 3D framework and model for self-supervised learning in medical imaging.*  
npj Digital Medicine.

DOI: https://doi.org/10.1038/s41746-025-02035-w

---

## Repository structure

```text
3D-HepaticNuclei-Classifier/
│
├── notebooks/
│   ├── dataset_helper.py
│   ├── 1_preprocessing.py
│   ├── 1_1_get_patches_sizes.py
│   ├── 1_1_get_patches_sizes_all.py
│   ├── 2_embedding_extraction.py
│   ├── 3_cross_validation_data.py
│   ├── 4_run_models.py
│   └── 5_run_CNN.py
│
├── statistic_notebooks/
│   ├── PCA.ipynb
│   ├── pca.py
│   └── boxplot_stadistic.py
│
├── LICENSE
├── LOG.md
└── README.md

