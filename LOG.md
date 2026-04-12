# Development Log (Research Diary)

**Project:** 3D-HepaticNuclei-Classifier  
**Author:** Alisson Terán  
**Thesis:** Automatic Classifier of Cell Nuclei in 3D Images of Liver Tissue using 3DINO
**Year:** 2026  

---
**Date:** 22/02/2026

## 1) Bounding Box Validation Issues

During the initial validation stage of the 3D patches, an inconsistency was detected in the bounding box generation process.  
The generated patches had size **12** instead of **13**, causing dimensional inconsistencies within the pipeline.

**Corrective actions:**
- Adjusted the bounding box configuration to generate patches of size **13**.
- Revalidated dimensional consistency.
- Created a dedicated validation notebook for bounding box verification.

---

## 2) Integration of the 3DINO Model from GitHub

One of the main technical challenges was integrating the 3DINO model from the official GitHub repository.

**Difficulties encountered:**
- Understanding the repository structure and dependencies.
- Correctly loading configuration files and pretrained weights.
- Managing environment setup within Google Colab and Google Drive.

**Solution implemented:**
- Explicit configuration of model paths.
- Proper loading of pretrained weights.
- Verification that the model runs in evaluation mode (`model.eval()`).
- Confirmation of correct GPU/CPU device allocation.

---

## 3) Data Preprocessing Challenges

Data preprocessing presented several uncertainties, mainly due to multiple possible strategies.

**Initial preprocessing steps included:**
- Normalization to range [-1, 1].
- Center padding and cropping.
- Fixed 3D resizing.
- Basic geometric augmentation.

**Identified difficulties:**
- Padding inconsistencies in certain cases.
- Potential noise introduced by normalization to [-1, 1].
- Uncertainty regarding optimal preprocessing strategy.

**Planned actions:**
- Test normalization in range [0, 1] for comparison.
- Re-evaluate padding consistency.
- Analyze impact on embedding separability.

---

## 4) Data Augmentation and Class Balancing

Current experiments were performed using aggressive class balancing through data augmentation, generating **7000 samples per class**.

However, no significant improvement was observed compared to the baseline model.

**Potential issues:**
- No systematic visual validation of augmentation transformations.
- Possible redundancy due to heavy oversampling.
- Limited generalization despite increased sample size.

**Planned experiments:**
- Run experiments without augmentation.
- Compare performance using:
  - 500 samples per class
  - 3500 samples per class
  - 7000 samples per class
    
- Incorporate controlled augmentation techniques:
  - Gaussian Noise
  - Gaussian Blur

---

## 5) Class Discrimination Difficulties

The results show consistent confusion among classes **1–2–3–4**, with particularly low recall for class 3.

This may indicate:

- True morphological overlap between cell types.
- Insufficiently discriminative latent embeddings.
- Domain mismatch between pretraining data and hepatic confocal volumes.

Additionally, hyperparameter optimization did not significantly outperform the baseline model, suggesting that performance limitations may be more related to representation quality than classifier tuning.

---

## 6) Proposed Ablation Study

If performance remains low after testing alternative preprocessing and augmentation strategies, the following experiment will be conducted:

**Action:**
- Exclude the heterogeneous "Other" class and retrain the model using only four main cell types.

**Objective:**
- Evaluate whether the inclusion of the "Other" class negatively impacts class separability and macro F1-score.

---

## 7) Next Steps

- Compare normalization ranges [-1, 1] vs [0, 1].
- Visually validate augmentation transformations.
- Conduct controlled experiments with varying dataset sizes.
- Evaluate Gaussian-based augmentation strategies.
- Perform class ablation if necessary.
- Explore embedding visualization (PCA / t-SNE).

---

## General Observation

So far, results suggest that performance limitations may be more strongly influenced by:

- Embedding quality.
- Data distribution and class overlap.
- Domain shift effects.

Rather than by hyperparameter optimization alone.

---

**Date:** 26/02/2026

**Objective:** Improve pipeline robustness and strengthen statistical validity.

### Embeddings v1 → v2

- Removed resize-based scaling (80³ → 112³) that introduced geometric distortion.
- Standardized input volume size to 112³ using pad/crop operations (no deformation).
- Added Gaussian noise as an independent augmentation transform.
- Implemented true 3D Gaussian Blur augmentation.
- Adjusted class balancing strategy: TARGET_PER_LABEL reduced from 7000 → 3500 (5 balanced classes).
- Added visualization and saving of augmented vs. non-augmented samples for quality control.
- Reduced debug print statements during data loading to prevent performance slowdown.

### Random Forest Training & Validation

- Fixed validation leakage during hyperparameter optimization (tuning performed only on Train set).
- Reserved hold-out validation strictly for final evaluation.
- Implemented group-aware splitting using file paths to prevent augmentation leakage.
- Maintained constrained hyperparameter search (n_iter=10, n_jobs=1) due to computational limitations.

### Verification

- Verified grouping by paths: 6958 unique paths / 17500 total samples.
- Confirmed no path intersection between Train and Validation after split (no leakage).

### General Observation

- Performance limitations appear more strongly influenced by:
  - Embedding quality
  - Data distribution and class overlap
  - Domain shift effects
- Rather than by hyperparameter optimization alone.


**Date:** 05/03/2026

**Objective:** Refine preprocessing, expand model development, and improve pipeline performance.

### Preprocessing and Embeddings

- Applied crop/pad operations to standardize nuclei volumes before the final input size.
- Tested preprocessing with crop/pad to **50³** as an intermediate step before resizing to **112³**.
- Evaluated the effect of this strategy on morphological preservation and embedding quality.
- Continued feature extraction using 3DINO-ViT under the updated preprocessing workflow.

### Model Development

- Started building the additional classification models for downstream evaluation:
  - Random Forest
  - Logistic Regression
  - Support Vector Machine
  - Multilayer Perceptron

### Hyperparameter Optimization

- Performed hyperparameter tuning under the revised preprocessing setting.
- Compared preprocessing alternatives in terms of their effect on downstream classification results.
- Maintained validation controls to ensure fair comparison between configurations.

### General Observation

- The preprocessing strategy had a direct effect on embedding quality and model performance.
- Preserving morphological consistency remained essential for robust classification.

---

**Date:** 19/03/2026

**Objective:** Evaluate model behavior without redundant balancing settings.

### Classification Configuration

- Tested the classification pipeline without applying the class-balancing parameter.
- Verified performance using already balanced image sets, avoiding unnecessary balancing during training.
- Assessed whether removing this parameter affected the stability and fairness of the models.

### Preprocessing and Training

- Continued experiments with the selected preprocessing configuration.
- Compared results obtained with and without the balancing parameter in order to determine its real contribution.
- Preserved the same validation strategy to ensure comparability.

### General Observation

- Since the dataset had already been balanced beforehand, the class-balancing parameter was no longer necessary.
- Removing this parameter simplified the configuration without negatively affecting the training setup.

---

**Date:** 26/03/2026

**Objective:** Consolidate intermediate results and advance the written analysis.

### Results Consolidation

- Advanced the drafting of the results section.
- Organized the main findings from preprocessing, embedding extraction, training, and validation.
- Integrated preliminary performance comparisons across experimental settings.

### Documentation

- Continued refining the description of the methodology and experimental pipeline.
- Structured the preliminary interpretation of the main metrics and observations.
- Consolidated the technical explanation of the improvements introduced across iterations.

### General Observation

- By this stage, the experimental results were sufficiently stable to support an initial written discussion.
- The work began to shift progressively from experimentation to analysis and reporting.

---

**Date:** 12/04/2026

**Objective:** Finalize the experimental stage and complete the final written results.

### Models Evaluated

- Obtained the complete results for all classification models:
  - Random Forest
  - Logistic Regression
  - Support Vector Machine
  - Multilayer Perceptron
- Included results for all augmentation settings and all hyperparameter optimization improvements.

### Final Adjustments

- Removed the class-balancing parameter from training because the classes had already been balanced beforehand.
- Standardized the final configuration used for model comparison.
- Consolidated the final evaluation setup for all selected classifiers.

### Documentation and Closure

- Finalized the writing of the experimental results section.
- Consolidated the final interpretation after completing all model comparisons.
- Closed the experimental stage with the complete integration of results, augmentations, and optimized configurations.

### General Observation

- The final comparison included all models, their augmentation variants, and their optimized hyperparameter settings.
- This enabled a complete and consistent interpretation of the experimental results.
