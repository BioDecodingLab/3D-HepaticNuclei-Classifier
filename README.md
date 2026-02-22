# 3D-HepaticNuclei-Classifier
A reproducible pipeline for 3D hepatic nucleus classification using self-supervised 3DINO embeddings and supervised machine learning.

## Overview

This repository implements a 3D nuclear morphotype classification pipeline based on:

3DINO self-supervised feature extraction
Bounding-box extraction of nuclear instances
Supervised classifiers (Random Forest, SVM, MLP)
3D microscopy datasets of mouse liver tissue

The goal is to evaluate how well self-supervised 3D embeddings support downstream classification of hepatic nuclear morphotypes.

This repository accompanies the thesis work:
“Automated Classification of Hepatic 3D Nuclear Morphotypes Using Self-Supervised 3DINO Embeddings” (2026)


## Key Features

* 3D nucleus preprocessing (bounding boxes, normalization)

* Self-supervised 3DINO embedding extraction

* Multiple classifier options (RF, SVM, MLP)

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


## Results (example)
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
