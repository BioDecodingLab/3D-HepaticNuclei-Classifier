# 3D-HepaticNuclei-Classifier
A reproducible pipeline for 3D hepatic nucleus classification using self-supervised 3DINO embeddings and supervised machine learning.

## Overview

This repository implements a 3D nuclear morphotype classification pipeline based on:

- **Self-supervised feature extraction:** 3DINO embeddings  
- **Nucleus instance preparation:** bounding-box (3D crop) extraction  
- **Downstream classifiers:** Random Forest (RF), Support Vector Machine (SVM), Multilayer Perceptron (MLP), Logistic Regression (LR)  
- **Data:** 3D microscopy volumes of mouse liver tissue

The goal is to evaluate how well self-supervised **3D embeddings** support downstream classification of **hepatic nuclear morphotypes**.

This repository accompanies the thesis work:  
**“Automated Classification of Hepatic 3D Nuclear Morphotypes Using Self-Supervised 3DINO Embeddings” (2026)**.


## Key Features

* 3D nucleus preprocessing (bounding boxes, normalization)

* Self-supervised 3DINO embedding extraction

* Multiple classifier options (RF, SVM, MLP, LR)

* Hyperparameter optimization experiments

* Evaluation metrics: accuracy, precision, recall, F1-score

* Configurable training pipeline

* Reproducible experiments using fixed seeds

* Lightweight and easy to extend for new tissues or modalities


## Current Research Notes

* Performance limitations observed between Stellate, Kupffer and Endothelial classes.

* Hyperparameter tuning of Random Forest did not significantly outperform baseline.

* Potential domain shift between 3DINO pretraining data and hepatic confocal volumes.

* Data augmentation strategies under review (visual verification pending).

* Ongoing evaluation of excluding heterogeneous “Other” class for ablation analysis.

## Running the project (Execution order)

The notebooks are numbered and must be executed in order:

## 🚀 Running the project (Execution order)

### **1. Patch size inspection**
**Files:**
- `1_1_get_patches_sizes.py`
- `1_1_get_patches_sizes_all.py`

**Description:**
- Inspects patch or bounding-box sizes in the 3D volumes  
- Helps define preprocessing parameters before standardization  

---

### **2. Data preprocessing**
**Files:**
- `dataset_helper.py`
- `1_preprocessing.py`

**Description:**
- Loads raw 3D volumes  
- Applies preprocessing such as normalization and crop/pad  
- Saves the processed dataset ready for embedding extraction  

---

### **3. Embedding extraction (3DINO / ViT features)**
**File:**
- `2_embedding_extraction.py`

**Description:**
- Loads the preprocessed volumes  
- Applies augmentations if configured  
- Extracts embeddings using 3DINO / Vision Transformer  
- Saves the resulting feature vectors  

---

### **4. Cross-validation split**
**File:**
- `3_cross_validation_data.py`

**Description:**
- Creates the train/validation/test folds  
- Organizes the evaluation setup for downstream classifiers  

---

### **5. Classical model training**
**File:**
- `4_run_models.py`

**Description:**
- Loads the embeddings  
- Trains and evaluates classical machine learning models such as:  
  - **Random Forest**  
  - **Support Vector Machine**  
  - **Logistic Regression**  

---

### **6. Statistics and visualization**
**File:**
- `5_box_plot_and_statistics_all_models.py`

**Description:**
- Summarizes model performance  
- Generates boxplots and statistical comparisons across models  

---

### **7. Interpretability analysis**
**File:**
- `6_shap_interpretability_test.py`

**Description:**
- Applies SHAP-based interpretability analysis  
- Identifies the most influential features for model predictions  

## Dependencies and versions

This project was tested on Google Colab (GPU recommended).
For the DINO embedding extraction notebook, install the following dependencies:

##Python environment

Python: Google Colab default (recommended)

GPU: CUDA-enabled runtime (recommended)

Packages (3DINO embeddings)
@article{xu3dino2025,
  title={A generalizable 3D framework and model for self-supervised learning in medical imaging},
  author={Xu, Tony and Hosseini, Sepehr and Anderson, Chris and Rinaldi, Anthony and Krishnan, Rahul G. and Martel, Anne L. and Goubran, Maged},
  journal={npj Digital Medicine},
  year={2025},
  doi={10.1038/s41746-025-02035-w},
}

## Repository Structure

3D-HepaticNuclei-Classifier/

├─ 📁 data/

│  ├─ 📁 raw/                # Raw microscopy volumes (not included)

│  ├─ 📁 processed/          # Crops/patches from bounding boxes

│  └─ 📁 embeddings/         # 3DINO embeddings (.npy)

├─ 📁 src/

│  ├─ 📁 preprocessing/      # Bounding-box extraction & 3D preprocessing

│  ├─ 📁 dino/               # 3DINO feature extraction

│  ├─ 📁 models/             # RF / SVM / MLP model definitions

│  ├─ 📁 training/           # Training & evaluation scripts

│  └─ 📁 utils/              # Metrics, plots, helpers

├─ 📁 notebooks/

│  ├─ 1_preprocessing.ipynb

│  ├─ 2_embedding_extraction.ipynb

│  ├─ 3_training_classifiers.ipynb

│  └─ 4_evaluation.ipynb

├─ 📁 results/

│  ├─ 📁 metrics/            # Scores, confusion matrices

│  └─ 📁 figures/            # Publication-ready plots

├─ environment.yml

├─ requirements.txt

├─ LICENSE

└─ README.md


## Dataset (Link to Zenodo dataset)

This project uses 3D confocal microscopy volumes of mouse liver tissue, provided by: Universidad de Concepción, Chile

Each nucleus is segmented via bounding boxes and annotated into classes such as:

1. Hepatocyte
2. Kupffer cell
3. Stellate cell
4. Endothelial cell
5. Other cell



## Getting Started

1. Install environment

conda env create -f environment.yml
conda activate hepatic-nuclei

2. Extract patches


3. Extract 3DINO embeddings


4. Train classifiers
  
5. Evaluate

6. Reaconstruct whole image


## Results 
Classifier	Accuracy	F1 (macro)	Notes

Random Forest	0.6739	0.5001	Fast, robust

SVM	

MLP	


## Citation

If you use this code, please cite:

@article{Teran2026Hepatic3DNuclei,
  title={Automated classification of hepatic 3D nuclear morphotypes using self-supervised 3D DINO embeddings},
  author={Terán Ballagán, A. C. and Morales-Navarrete, H. A.},
  year={2026},
  journal={IEEE Access}
}
