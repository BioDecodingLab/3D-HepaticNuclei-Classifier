"""
Create leave-one-image-out cross-validation .npz files from precomputed embeddings.

Expected input structure
------------------------
aug_dir/
    img_1_2000_samples_aug.npz
    img_2_2000_samples_aug.npz
    img_3_2000_samples_aug.npz
    img_4_2000_samples_aug.npz
    img_5_2000_samples_aug.npz

test_dir/
    img_1.npz
    img_2.npz
    img_3.npz
    img_4.npz
    img_5.npz

Each source .npz is expected to contain at least:
    - embeddings
    - labels
Optionally:
    - paths

Output
------
For each fold, saves one .npz file with:
    - X_train, y_train
    - X_val,   y_val
    - X_test,  y_test

Additionally stores:
    - paths_train, paths_val, paths_test
    - train_subject_ids, val_subject_ids, test_subject_id

Example fold:
    train = img_1_aug + img_2_aug + img_3_aug + img_4_aug
    val   = 10% split from train
    test  = img_5

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split


import time

#print("Pausing for 1 hour...")
#time.sleep(4800)  # 3600 seconds = 1 hour
#print("Resuming...")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
AUG_DIR = Path("/medicina/hmorales/projects/Nuclei3DClassification/data/embedding_aug_2000")   #_aug_500
TEST_DIR = Path("/medicina/hmorales/projects/Nuclei3DClassification/data/embedding")
OUT_DIR = Path("/medicina/hmorales/projects/Nuclei3DClassification/data/embedding_2000_cv")  #_500_cv
TRAIN_SUFFIX = "_2000_samples_aug"

RANDOM_STATE = 42
VAL_FRACTION = 0.10
SUBJECT_IDS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_embedding_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Load one embedding .npz file.

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file.

    Returns
    -------
    dict
        Dictionary with keys:
        - embeddings : np.ndarray, shape (N, D)
        - labels     : np.ndarray, shape (N,)
        - paths      : np.ndarray, shape (N,), dtype=str
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    required = {"embeddings", "labels"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{npz_path} is missing required keys: {sorted(missing)}")

    embeddings = data["embeddings"]
    labels = data["labels"]

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mismatch in {npz_path}: embeddings has {embeddings.shape[0]} samples, "
            f"labels has {labels.shape[0]} samples."
        )

    if "paths" in data.files:
        paths = data["paths"].astype(str)
        if len(paths) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch in {npz_path}: paths has {len(paths)} entries, "
                f"but embeddings has {embeddings.shape[0]} samples."
            )
    else:
        paths = np.array([f"{npz_path.stem}__sample_{i}" for i in range(len(labels))], dtype=str)

    return {
        "embeddings": embeddings,
        "labels": labels,
        "paths": paths,
    }


def concatenate_blocks(blocks: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate multiple embedding blocks.

    Parameters
    ----------
    blocks : list of dict
        Each dict contains embeddings, labels, and paths.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    paths : np.ndarray
    """
    if len(blocks) == 0:
        raise ValueError("No blocks provided for concatenation.")

    X = np.concatenate([b["embeddings"] for b in blocks], axis=0)
    y = np.concatenate([b["labels"] for b in blocks], axis=0)
    paths = np.concatenate([b["paths"] for b in blocks], axis=0)

    if not (len(X) == len(y) == len(paths)):
        raise RuntimeError("Concatenation produced inconsistent sample counts.")

    return X, y, paths


def is_stratification_possible(y: np.ndarray, val_fraction: float) -> bool:
    """
    Check whether a stratified split is feasible.

    Stratified split can fail if some classes are too small.

    Parameters
    ----------
    y : np.ndarray
        Labels.
    val_fraction : float
        Fraction for validation split.

    Returns
    -------
    bool
    """
    unique, counts = np.unique(y, return_counts=True)

    if len(unique) < 2:
        return False

    # Very conservative safeguard:
    # require at least 2 samples per class
    if np.any(counts < 2):
        return False

    # Also ensure validation set can contain at least one sample for multiple classes
    n_val = max(1, int(round(len(y) * val_fraction)))
    if n_val < len(unique):
        return False

    return True


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    paths: np.ndarray,
    val_fraction: float = 0.10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data into train/validation.

    Uses stratification if feasible, otherwise falls back to random split.

    Returns
    -------
    X_train, X_val, y_train, y_val, paths_train, paths_val
    """
    stratify = y if is_stratification_possible(y, val_fraction) else None

    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
        X,
        y,
        paths,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    return X_train, X_val, y_train, y_val, paths_train, paths_val


def save_fold_npz(
    out_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    paths_train: np.ndarray,
    paths_val: np.ndarray,
    paths_test: np.ndarray,
    train_subject_ids: List[int],
    val_subject_ids: List[int],
    test_subject_id: int,
) -> None:
    """
    Save one fold to compressed .npz.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        paths_train=paths_train.astype(str),
        paths_val=paths_val.astype(str),
        paths_test=paths_test.astype(str),
        train_subject_ids=np.asarray(train_subject_ids, dtype=int),
        val_subject_ids=np.asarray(val_subject_ids, dtype=int),
        test_subject_id=np.asarray([test_subject_id], dtype=int),
    )


def describe_labels(name: str, y: np.ndarray) -> str:
    """
    Create a short class-distribution string.
    """
    unique, counts = np.unique(y, return_counts=True)
    parts = [f"{cls}:{cnt}" for cls, cnt in zip(unique, counts)]
    return f"{name} n={len(y)} | classes {{{', '.join(parts)}}}"


# ---------------------------------------------------------------------
# Main CV generation
# ---------------------------------------------------------------------
def create_leave_one_image_out_cv(
    aug_dir: Path,
    test_dir: Path,
    out_dir: Path,
    subject_ids: List[int],
    val_fraction: float = 0.10,
    random_state: int = 42,
    train_suffix: str = "2000_samples_aug", 
) -> None:
    """
    Create leave-one-image-out cross-validation files.

    For each held-out subject k:
      - training pool = all augmented files except k
      - test set      = non-augmented file k
      - validation    = 10% of training pool

    Parameters
    ----------
    aug_dir : Path
        Directory with augmented training embeddings.
    test_dir : Path
        Directory with non-augmented embeddings used as held-out test sets.
    out_dir : Path
        Output directory for fold .npz files.
    subject_ids : list of int
        Subject/image IDs.
    val_fraction : float
        Validation fraction from training pool.
    random_state : int
        Seed for reproducible splitting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for test_id in subject_ids:
        print("=" * 90)
        print(f"Creating fold with held-out test image: img_{test_id}")

        train_ids = [sid for sid in subject_ids if sid != test_id]

        # Load training pool from augmented files
        train_blocks = []
        for sid in train_ids:
            aug_file = aug_dir / f"img_{sid}{train_suffix}.npz"
            train_blocks.append(load_embedding_npz(aug_file))

        # Load test set from non-augmented file
        test_file = test_dir / f"img_{test_id}.npz"
        test_block = load_embedding_npz(test_file)

        # Concatenate all train blocks
        X_pool, y_pool, paths_pool = concatenate_blocks(train_blocks)

        # Test block
        X_test = test_block["embeddings"]
        y_test = test_block["labels"]
        paths_test = test_block["paths"]

        # Split train/val
        X_train, X_val, y_train, y_val, paths_train, paths_val = split_train_val(
            X_pool,
            y_pool,
            paths_pool,
            val_fraction=val_fraction,
            random_state=random_state,
        )

        # Save fold
        out_path = out_dir / f"cv_fold_test_img_{test_id}.npz"
        save_fold_npz(
            out_path=out_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            paths_train=paths_train,
            paths_val=paths_val,
            paths_test=paths_test,
            train_subject_ids=train_ids,
            val_subject_ids=train_ids,   # validation comes from training pool
            test_subject_id=test_id,
        )

        # Summary
        print(describe_labels("TRAIN", y_train))
        print(describe_labels("VAL  ", y_val))
        print(describe_labels("TEST ", y_test))
        print(f"Saved: {out_path}")

    print("=" * 90)
    print(f"Done. Cross-validation files saved in: {out_dir.resolve()}")


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    create_leave_one_image_out_cv(
        aug_dir=AUG_DIR,
        test_dir=TEST_DIR,
        out_dir=OUT_DIR,
        subject_ids=SUBJECT_IDS,
        val_fraction=VAL_FRACTION,
        random_state=RANDOM_STATE,
        train_suffix = TRAIN_SUFFIX,
    )