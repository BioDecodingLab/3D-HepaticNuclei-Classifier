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

Cell-type-specific fluorescence channels were used for ground-truth annotation, whereas the computational classification experiments used only DAPI-derived nuclear information as model input.

## Experimental design

The dataset consisted of five 3D mouse liver images, each obtained from a different animal.

Model evaluation was performed using leave-one-image/animal-out cross-validation, resulting in five independent test folds.

For the 3DINO and handcrafted-feature approaches, class-balanced augmented training datasets containing 100, 500, 1000, 2000, or 4000 samples per class were evaluated.

The classical classifiers included:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Multilayer Perceptron (MLP)

Macro-F1 was used as the primary model-selection metric because it gives equal weight to all five nuclear morphotype classes.

## Key features

- 3D nuclear patch extraction and preprocessing
- DAPI-only classification
- Frozen 3DINO embedding extraction
- Handcrafted 3D nuclear feature analysis
- Direct ResNet3D-18 patch classification
- Leave-one-image/animal-out cross-validation
- Class-balanced data augmentation
- Logistic Regression, Random Forest, SVM, and MLP classifiers
- PCA-based representation analysis
- Evaluation using macro-F1, weighted-F1, balanced accuracy, accuracy, precision, recall, and ROC-AUC
- Friedman tests and average-rank analysis for model comparison
- Paired Wilcoxon signed-rank tests with Holm correction
- Paired Cohen's d effect sizes
- Reproducible experiments using fixed random seeds

## Running the project

### 1. Data preprocessing

**Files:**
- `notebooks/dataset_helper.py`
- `notebooks/1_preprocessing.py`

The preprocessing pipeline loads individual 3D nuclear patches, applies intensity normalization and spatial standardization, and prepares the data for subsequent representation extraction.

### 1.1 Patch size inspection

**Files:**
- `notebooks/1_1_get_patches_sizes.py`
- `notebooks/1_1_get_patches_sizes_all.py`

These scripts inspect the original 3D nuclear bounding-box dimensions and support the definition of the standardized patch size used during preprocessing.

### 2. 3DINO embedding extraction

**File:**
- `notebooks/2_embedding_extraction.py`

This script:

- loads preprocessed 3D DAPI nuclear patches,
- applies training augmentation when configured,
- extracts frozen 3DINO embeddings,
- and saves the resulting feature vectors.

### 3. Cross-validation split generation

**File:**
- `notebooks/3_cross_validation_data.py`

This script generates the five leave-one-image/animal-out folds.

For each fold:

- one animal/image is used exclusively for testing,
- the remaining four animals/images form the training pool,
- and 10% of the training data are used for validation.

The non-augmented data from the held-out animal are retained for testing.

### 4. Classical machine-learning models

**File:**
- `notebooks/4_run_models.py`

The following classifiers are evaluated on the extracted representations:

- Logistic Regression
- Random Forest
- Support Vector Machine
- Multilayer Perceptron

Model selection and hyperparameter optimization are performed using the training and validation data within each fold.

### 5. Statistical analysis and model comparison

**File:**
- `statistic_notebooks/Boxplot_stadistic.ipynb`

This notebook performs the final statistical analysis used for the study, including:

- aggregation of test-fold metrics,
- comparison of augmentation levels and classifiers,
- Friedman tests,
- average model ranks,
- paired Wilcoxon signed-rank tests,
- Holm correction for multiple comparisons,
- paired Cohen's d effect sizes,
- comparison of the selected 3DINO, handcrafted-feature, and ResNet3D-18 models,
- and generation of the performance boxplots.

### 6. PCA analysis

**File:**
- `statistic_notebooks/PCA.ipynb`

This notebook performs principal component analysis of:

- handcrafted DAPI-derived nuclear features, and
- frozen 3DINO embeddings.

The PCA analysis is used to visualize the organization of nuclear representations across hepatic morphotype classes and animals/images.

## Results

The best-performing configuration from each methodological family was:

| Feature set / approach | Selected model | Accuracy | Weighted-F1 | Macro-F1 | Balanced Accuracy | AUC Macro OVR |
|---|---|---:|---:|---:|---:|---:|
| 3DINO embeddings | SVM + 2000 | 0.8333 | 0.8409 | 0.6132 | 0.6417 | 0.9272 |
| Direct patch classification | ResNet3D-18 | 0.8326 | 0.8228 | 0.5352 | 0.5360 | — |
| Handcrafted features | RF + 4000 | 0.7626 | 0.7753 | 0.5157 | 0.5521 | 0.9037 |

The 3DINO + SVM model trained using 2000 augmented samples per class achieved the highest macro-F1 and balanced accuracy among the three selected methodological approaches.

Across the five independent animal-level test folds, its performance was:

- **Macro-F1:** 0.613 ± 0.043
- **Weighted-F1:** 0.841 ± 0.037
- **Balanced accuracy:** 0.642 ± 0.051

Fold-wise improvements and paired effect sizes favored the 3DINO-based approach. However, pairwise Wilcoxon signed-rank comparisons did not remain statistically significant after Holm correction, consistent with the limited number of independent animal-level test folds.

## Dataset

The study uses 3D fluorescence microscopy volumes of mouse liver tissue provided by Universidad de Concepción, Chile.

The data supporting this study are publicly available on Zenodo:

https://doi.org/10.5281/zenodo.19502784

## 3DINO reference

The self-supervised embeddings were generated using the 3DINO framework:

Xu, T., Hosseini, S., Anderson, C., Rinaldi, A., Krishnan, R. G., Martel, A. L., & Goubran, M. (2025). A generalizable 3D framework and model for self-supervised learning in medical imaging. *npj Digital Medicine*.

DOI: https://doi.org/10.1038/s41746-025-02035-w

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
│   └── 5_box_plot_and_statistics_all_models.py
│
├── statistic_notebooks/
│   ├── Boxplot_stadistic.ipynb
│   └── PCA.ipynb
│
├── LICENSE
├── LOG.md
└── README.md

