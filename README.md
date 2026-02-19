# 3D-HepaticNuclei-Classifier
A lightweight and reproducible pipeline for 3D nucleus classification in hepatic tissue using 3D self-supervised embeddings (3DINO) and supervised machine learning.

📌 Overview

This repository contains the implementation of a 3D nucleus classification pipeline built on:

3DINO self-supervised representations
Bounding-box extraction of nuclear instances
Supervised classifiers (Random Forest, SVM, MLP)
3D microscopy datasets of mouse liver tissue

The goal is to provide an efficient, scalable, and reproducible workflow for the automatic classification of hepatic nuclear morphotypes from volumetric images.

This codebase accompanies the work:
“Automated Classification of Hepatic 3D Nuclear Morphotypes Using Self-Supervised 3D DINO Embeddings” (2026).


✨ Key Features

✔ 3D nucleus preprocessing (bounding boxes, normalization)
✔ Self-supervised 3DINO embedding extraction
✔ Multiple classifier options (RF, SVM, MLP)
✔ Evaluation metrics: accuracy, precision, recall, F1-score
✔ Configurable training pipeline
✔ Reproducible experiments using fixed seeds
✔ Lightweight and easy to extend for new tissues or modalities




📁 Repository Structure

3D-HepaticNuclei-Classifier/

│

├── data/

│   ├── raw/                 # Microscopy raw data (not included)

│   ├── processed/           # Bounding boxes, normalized crops

│   └── embeddings/          # 3DINO embeddings (.npy)

│

├── src/

│   ├── preprocessing/       # Bounding-box extraction & 3D preprocessing

│   ├── dino/                # 3DINO feature extraction scripts

│   ├── models/              # RF, SVM, MLP models

│   ├── utils/               # Helpers, metrics, visualization

│   └── training/            # Training & evaluation loops

│

├── notebooks/

│   ├── 1_preprocessing.ipynb

│   ├── 2_embedding_extraction.ipynb

│   ├── 3_training_classifiers.ipynb

│   └── 4_evaluation.ipynb

│

├── results/

│   ├── metrics/             # F1, confusion matrices

│   └── figures/             # Visualizations for publication

│

├── environment.yml          # Pixi / Conda environment file

├── requirements.txt         # Alternative Python dependency list

├── LICENSE

└── README.md



🧬 Dataset (Link to Zenodo dataset)

This project uses 3D confocal microscopy volumes of mouse liver tissue, provided by: Universidad de Concepción, Chile

Each nucleus is segmented via bounding boxes and annotated into classes such as:

-Hepatocyte
-Kupffer cell
-Stellate cell
-Endothelial cell
-Other cell



🚀 Getting Started
1. Install environment

conda env create -f environment.yml
conda activate hepatic-nuclei

2. Extract patches


3. Extract 3DINO embeddings


4. Train classifiers
  
5. Evaluate

6. Reaconstruct whole image


📊 Results (example)
Classifier	Accuracy	F1 (macro)	Notes
Random Forest	0.92	0.89	Fast, robust
SVM	0.88	0.86	Sensitive to scaling
MLP	0.94	0.91	Best overall


📝 Citation

If you use this code, please cite:

@article{Teran2026Hepatic3DNuclei,
  title={Automated classification of hepatic 3D nuclear morphotypes using self-supervised 3D DINO embeddings},
  author={Terán Ballagán, A. C. and Morales-Navarrete, H. A.},
  year={2026},
  journal={IEEE Access}
}
