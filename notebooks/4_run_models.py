# -*- coding: utf-8 -*-
"""
Cross-validation benchmark script with:
- hyperparameter search per fold
- final refit on train+val
- ROC curves per class (OvR)
- pooled ROC across folds
- explicit saving of best refit model per fold
"""

import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time

#print("Pausing for 1 hour...")
#time.sleep(6000)  # 3600 seconds = 1 hour
#print("Resuming...")

# ============================================================
# USER CONFIG
# ============================================================

INPUT_DIR = "/medicina/hmorales/projects/Nuclei3DClassification/data/embedding_2000_cv"
OUTPUT_ROOT = "/medicina/hmorales/projects/Nuclei3DClassification/results/cv_benchmark_results_2000_cv"

LABEL_ORDER = [1, 2, 3, 4, 5]
CLASS_NAMES = ["Hepatocyte", "Stellate", "Kupffer", "Endotelial", "Other"]

RANDOM_STATE = 42
MODEL_NAMES = ["logreg", "rf", "svm", "mlp"]  # choose from: "logreg", "rf", "svm", "mlp"

# Parallelism:
# This parallelizes hyperparameter combinations WITHIN each fold.
# Folds themselves are processed sequentially to avoid CPU oversubscription.
N_JOBS_SEARCH = 48
VERBOSE_SEARCH = 10

# Whether to save train/val diagnostic metrics per fold
SAVE_DIAGNOSTIC_TRAIN_VAL = True


# ============================================================
# PARAMETER GRIDS
# ============================================================

PARAM_GRIDS = {
    "logreg": {
        "pca__n_components": [0.95, 0.5],
        "pca__whiten": [True, False],
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 50],
        "clf__penalty": ["l2"],
        "clf__class_weight": [None],
    },
    "rf": {
        "pca__n_components": [0.95, 0.5],
        "pca__whiten": [False],
        "clf__n_estimators": [200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [5, 10],
        "clf__min_samples_leaf": [2, 4],
        "clf__max_features": ["sqrt"],
        "clf__class_weight": [None],
    },
    "svm": {
        "pca__n_components": [0.95, 0.5],
        "pca__whiten": [True, False],
        "clf__C": [0.001, 0.01, 0.1, 1, 10],
        "clf__kernel": ["rbf"],
        "clf__gamma": ["scale", 1e-3],
        "clf__class_weight": [None],
    },
    "mlp": {
        "pca__n_components": [0.95, 0.5],
        "pca__whiten": [False],
        "clf__hidden_layer_sizes": [
            (256, 128),
            (512, 256),
            (256, 128, 64)
        ],
        "clf__alpha": [1e-4, 1e-3],
        "clf__learning_rate_init": [1e-3, 5e-4],
        "clf__batch_size": [64, 128]
    }
}


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def print_class_counts(y, split_name):
    counts = pd.Series(y).value_counts().sort_index()
    print(f"{split_name} counts:", counts.to_dict())


def filter_valid_pca_components(param_grid, X_train):
    """
    Keep only valid PCA integer components.
    Floats like 0.95 are always kept.
    """
    filtered = dict(param_grid)
    max_valid = min(X_train.shape[0], X_train.shape[1])

    valid_components = []
    for val in filtered["pca__n_components"]:
        if isinstance(val, float):
            if 0 < val < 1:
                valid_components.append(val)  # variance-based PCA
            elif val.is_integer() and val <= max_valid:
                valid_components.append(int(val))
        elif isinstance(val, int) and val <= max_valid:
            valid_components.append(val)

    if len(valid_components) == 0:
        raise ValueError(
            f"No valid PCA components remain. "
            f"Requested={filtered['pca__n_components']}, max_valid={max_valid}"
        )

    filtered["pca__n_components"] = valid_components
    return filtered


def serialize_best_params(best_row):
    """
    Extract hyperparameters from best_row and fix pandas dtype coercions.
    """
    best_params = {}
    for k, v in best_row.items():
        if k.startswith("pca__") or k.startswith("clf__"):
            best_params[k] = v

    # RF: max_depth may come back as float or NaN after pandas
    if "clf__max_depth" in best_params:
        v = best_params["clf__max_depth"]
        if pd.isna(v):
            best_params["clf__max_depth"] = None
        else:
            best_params["clf__max_depth"] = int(v)

    # RF integer params
    for k in ["clf__n_estimators", "clf__min_samples_split", "clf__min_samples_leaf"]:
        if k in best_params and not pd.isna(best_params[k]):
            best_params[k] = int(best_params[k])

    # MLP batch size
    if "clf__batch_size" in best_params and not pd.isna(best_params["clf__batch_size"]):
        best_params["clf__batch_size"] = int(best_params["clf__batch_size"])

    # PCA integer components, but keep variance fractions like 0.95 / 0.5
    if "pca__n_components" in best_params:
        v = best_params["pca__n_components"]
        if isinstance(v, float) and v >= 1 and float(v).is_integer():
            best_params["pca__n_components"] = int(v)

    return best_params


def build_pipeline(model_name, params, random_state=42):
    pca = PCA(
        n_components=params["pca__n_components"],
        whiten=params["pca__whiten"],
        random_state=random_state
    )

    if model_name == "logreg":
        clf = LogisticRegression(
            C=params["clf__C"],
            penalty=params["clf__penalty"],
            solver="lbfgs",
            class_weight=params["clf__class_weight"],
            max_iter=5000,
            random_state=random_state,
            multi_class="multinomial"
        )

    elif model_name == "rf":
        max_depth = params["clf__max_depth"]
        if pd.isna(max_depth):
            max_depth = None
        elif max_depth is not None:
            max_depth = int(max_depth)

        clf = RandomForestClassifier(
            n_estimators=int(params["clf__n_estimators"]),
            max_depth=max_depth,
            min_samples_split=int(params["clf__min_samples_split"]),
            min_samples_leaf=int(params["clf__min_samples_leaf"]),
            max_features=params["clf__max_features"],
            class_weight=params["clf__class_weight"],
            random_state=random_state,
            bootstrap=True,
            n_jobs=1
        )

    elif model_name == "svm":
        clf = SVC(
            C=params["clf__C"],
            kernel=params["clf__kernel"],
            gamma=params["clf__gamma"],
            class_weight=params["clf__class_weight"],
            probability=True,
            random_state=random_state
        )

    elif model_name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=params["clf__hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=params["clf__alpha"],
            batch_size=int(params["clf__batch_size"]),
            learning_rate_init=params["clf__learning_rate_init"],
            max_iter=500,
            shuffle=True,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=False
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", pca),
        ("clf", clf)
    ])


def compute_metrics(y_true, y_pred, split_name):
    overall = {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
    }

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    report_txt = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    per_class_rows = []
    for cname in CLASS_NAMES:
        per_class_rows.append({
            "split": split_name,
            "class_name": cname,
            "precision": report_dict[cname]["precision"],
            "recall": report_dict[cname]["recall"],
            "f1_score": report_dict[cname]["f1-score"],
            "support": report_dict[cname]["support"]
        })

    per_class_df = pd.DataFrame(per_class_rows)
    return overall, per_class_df, report_txt


def save_confusion_matrix(y_true, y_pred, save_path, title, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER, normalize=normalize)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    values_format = ".2f" if normalize is not None else "d"
    disp.plot(ax=ax, cmap="Blues", values_format=values_format, xticks_rotation=45, colorbar=False)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path,  bbox_inches="tight")
    #plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics_table_png(metrics_df, save_path, title):
    df_show = metrics_df.copy().round(4)

    fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(df_show)))
    ax.axis("off")

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title(title, pad=20)
    plt.tight_layout()
    #plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def get_model_scores(model, X):
    """
    Return class scores for ROC/AUC.
    Prefer predict_proba; fall back to decision_function if needed.
    Output shape: (n_samples, n_classes)
    """
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
        if y_score.ndim == 1:
            y_score = y_score[:, np.newaxis]
    else:
        y_score = None
    return y_score


def compute_multiclass_roc(y_true, y_score):
    """
    Compute one-vs-rest ROC curves and macro AUC.
    """
    if y_score is None:
        return None, None, None, np.nan

    y_true_bin = label_binarize(y_true, classes=LABEL_ORDER)

    if y_score.shape[1] != len(LABEL_ORDER):
        raise ValueError(
            f"y_score has shape {y_score.shape}, expected second dimension = {len(LABEL_ORDER)}"
        )

    fpr = {}
    tpr = {}
    auc_per_class = {}

    for i in range(len(CLASS_NAMES)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        auc_per_class[i] = roc_auc_score(y_true_bin[:, i], y_score[:, i])

    auc_macro = roc_auc_score(
        y_true_bin,
        y_score,
        multi_class="ovr",
        average="macro"
    )

    return fpr, tpr, auc_per_class, auc_macro


def save_roc_curves(y_true, y_score, save_path, title):
    """
    Save per-class ROC curves and return macro AUC.
    """
    fpr, tpr, auc_per_class, auc_macro = compute_multiclass_roc(y_true, y_score)

    if fpr is None:
        return np.nan

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(CLASS_NAMES):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"{class_name} (AUC = {auc_per_class[i]:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title + f"\nMacro AUC-ROC (OvR) = {auc_macro:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return auc_macro


def score_model_on_validation(model_name, params, X_train, y_train, X_val, y_val, random_state=42):
    model = build_pipeline(model_name, params, random_state=random_state)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    f1_weighted = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_val, y_val_pred)

    if hasattr(model, "predict_proba"):
        y_val_proba = model.predict_proba(X_val)
        ll = log_loss(y_val, y_val_proba, labels=LABEL_ORDER)
    else:
        ll = np.nan

    result = dict(params)
    result.update({
        "accuracy_val": acc,
        "f1_weighted_val": f1_weighted,
        "f1_macro_val": f1_macro,
        "balanced_acc_val": bal_acc,
        "log_loss_val": ll
    })
    return result


def select_best_result(results_df):
    """
    Selection criterion:
    1) highest macro F1
    2) highest balanced accuracy
    3) lowest log-loss if available
    """
    df = results_df.copy()
    df["_log_loss_fill"] = df["log_loss_val"].fillna(np.inf)

    df = df.sort_values(
        by=["f1_macro_val", "balanced_acc_val", "_log_loss_fill"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    best_row = df.iloc[0].to_dict()
    df = df.drop(columns=["_log_loss_fill"])
    return best_row, df


def run_hyperparameter_search(model_name, param_grid, X_train, y_train, X_val, y_val, n_jobs=48, random_state=42):
    param_list = list(ParameterGrid(param_grid))
    print(f"Total hyperparameter combinations: {len(param_list)}")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=VERBOSE_SEARCH)(
        delayed(score_model_on_validation)(
            model_name, params, X_train, y_train, X_val, y_val, random_state
        )
        for params in param_list
    )

    results_df = pd.DataFrame(results)
    best_row, sorted_df = select_best_result(results_df)
    return best_row, sorted_df


def fit_selected_model_and_evaluate(
    model_name,
    best_params,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    random_state=42,
    save_diagnostic_train_val=True
):
    outputs = {}

    # Diagnostic model: train only
    if save_diagnostic_train_val:
        diag_model = build_pipeline(model_name, best_params, random_state=random_state)
        diag_model.fit(X_train, y_train)

        y_train_pred = diag_model.predict(X_train)
        y_val_pred = diag_model.predict(X_val)

        y_train_score = get_model_scores(diag_model, X_train)
        y_val_score = get_model_scores(diag_model, X_val)

        train_metrics, train_per_class, train_report = compute_metrics(y_train, y_train_pred, "Train")
        val_metrics, val_per_class, val_report = compute_metrics(y_val, y_val_pred, "Validation")

        _, _, _, train_auc_macro = compute_multiclass_roc(y_train, y_train_score)
        _, _, _, val_auc_macro = compute_multiclass_roc(y_val, y_val_score)

        train_metrics["roc_auc_macro_ovr"] = train_auc_macro
        val_metrics["roc_auc_macro_ovr"] = val_auc_macro

        outputs["diagnostic_model"] = diag_model
        outputs["train_metrics"] = train_metrics
        outputs["val_metrics"] = val_metrics
        outputs["train_per_class"] = train_per_class
        outputs["val_per_class"] = val_per_class
        outputs["train_report"] = train_report
        outputs["val_report"] = val_report
        outputs["y_train_pred"] = y_train_pred
        outputs["y_val_pred"] = y_val_pred
        outputs["y_train_score"] = y_train_score
        outputs["y_val_score"] = y_val_score

    # Final model: refit on train + val
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    final_model = build_pipeline(model_name, best_params, random_state=random_state)
    final_model.fit(X_trainval, y_trainval)

    y_test_pred = final_model.predict(X_test)
    y_test_score = get_model_scores(final_model, X_test)

    test_metrics, test_per_class, test_report = compute_metrics(y_test, y_test_pred, "Test")
    _, _, _, test_auc_macro = compute_multiclass_roc(y_test, y_test_score)
    test_metrics["roc_auc_macro_ovr"] = test_auc_macro

    outputs["final_model"] = final_model
    outputs["best_model"] = final_model
    outputs["test_metrics"] = test_metrics
    outputs["test_per_class"] = test_per_class
    outputs["test_report"] = test_report
    outputs["y_test_pred"] = y_test_pred
    outputs["y_test_true"] = y_test
    outputs["y_test_score"] = y_test_score

    return outputs


# ============================================================
# PER-FOLD PROCESSING
# ============================================================

def process_single_fold(npz_path, output_root, model_name, param_grid, random_state=42):
    npz_path = Path(npz_path)
    fold_name = npz_path.stem

    fold_output_dir = Path(output_root) / model_name / fold_name
    ensure_dir(fold_output_dir)

    print("\n" + "=" * 100)
    print(f"Processing fold: {fold_name}")
    print(f"File: {npz_path.name}")
    print(f"Model: {model_name}")
    print("=" * 100)

    data = np.load(npz_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    print("Classes in train:", np.unique(y_train))
    print("Classes in val  :", np.unique(y_val))
    print("Classes in test :", np.unique(y_test))

    print_class_counts(y_train, "Train")
    print_class_counts(y_val, "Validation")
    print_class_counts(y_test, "Test")

    param_grid_local = filter_valid_pca_components(param_grid, X_train)

    # Hyperparameter search
    best_row, results_df = run_hyperparameter_search(
        model_name=model_name,
        param_grid=param_grid_local,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_jobs=N_JOBS_SEARCH,
        random_state=random_state
    )

    best_params = serialize_best_params(best_row)

    print("\nBest hyperparameters:")
    print(json.dumps(best_params, indent=2, default=str))

    print("\nBest hyperparameters with types:")
    for k, v in best_params.items():
        print(f"{k}: {v} ({type(v)})")

    print(f"Best validation macro F1: {best_row['f1_macro_val']:.6f}")
    print(f"Best validation balanced accuracy: {best_row['balanced_acc_val']:.6f}")
    print(f"Best validation log-loss: {best_row['log_loss_val']}")

    # Save search results
    results_csv_path = fold_output_dir / "hyperparameter_search_results.csv"
    best_params_json_path = fold_output_dir / "best_params.json"
    final_model_path = fold_output_dir / "final_model.joblib"
    best_model_path = fold_output_dir / "best_model_trainval.joblib"
    metrics_csv_path = fold_output_dir / "metrics_summary.csv"
    per_class_csv_path = fold_output_dir / "per_class_metrics.csv"
    reports_txt_path = fold_output_dir / "classification_reports.txt"
    metrics_png_path = fold_output_dir / "metrics_summary.svg"

    cm_train_raw_path = fold_output_dir / "cm_train_raw.svg"
    cm_train_norm_path = fold_output_dir / "cm_train_normalized.svg"
    cm_val_raw_path = fold_output_dir / "cm_val_raw.svg"
    cm_val_norm_path = fold_output_dir / "cm_val_normalized.svg"
    cm_test_raw_path = fold_output_dir / "cm_test_raw.svg"
    cm_test_norm_path = fold_output_dir / "cm_test_normalized.svg"

    roc_train_path = fold_output_dir / "roc_train_ovr.svg"
    roc_val_path = fold_output_dir / "roc_val_ovr.svg"
    roc_test_path = fold_output_dir / "roc_test_ovr.svg"

    results_df.to_csv(results_csv_path, index=False)

    with open(best_params_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "fold_name": fold_name,
            "file_name": npz_path.name,
            "model_name": model_name,
            "best_params": best_params,
            "best_validation_metrics": {
                "accuracy_val": float(best_row["accuracy_val"]),
                "f1_weighted_val": float(best_row["f1_weighted_val"]),
                "f1_macro_val": float(best_row["f1_macro_val"]),
                "balanced_acc_val": float(best_row["balanced_acc_val"]),
                "log_loss_val": None if pd.isna(best_row["log_loss_val"]) else float(best_row["log_loss_val"]),
            }
        }, f, indent=2, ensure_ascii=False, default=str)

    # Fit selected model and evaluate
    eval_out = fit_selected_model_and_evaluate(
        model_name=model_name,
        best_params=best_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state,
        save_diagnostic_train_val=SAVE_DIAGNOSTIC_TRAIN_VAL
    )

    metrics_rows = []
    per_class_parts = []
    report_blocks = []

    if SAVE_DIAGNOSTIC_TRAIN_VAL:
        metrics_rows.append(eval_out["train_metrics"])
        metrics_rows.append(eval_out["val_metrics"])
        per_class_parts.append(eval_out["train_per_class"])
        per_class_parts.append(eval_out["val_per_class"])

        report_blocks.append("CLASSIFICATION REPORT - TRAIN\n" + "-" * 80 + "\n" + eval_out["train_report"])
        report_blocks.append("CLASSIFICATION REPORT - VALIDATION\n" + "-" * 80 + "\n" + eval_out["val_report"])

        save_confusion_matrix(
            y_train, eval_out["y_train_pred"],
            cm_train_raw_path,
            f"{fold_name} - Train (raw)",
            normalize=None
        )
        save_confusion_matrix(
            y_train, eval_out["y_train_pred"],
            cm_train_norm_path,
            f"{fold_name} - Train (normalized)",
            normalize="true"
        )
        save_confusion_matrix(
            y_val, eval_out["y_val_pred"],
            cm_val_raw_path,
            f"{fold_name} - Validation (raw)",
            normalize=None
        )
        save_confusion_matrix(
            y_val, eval_out["y_val_pred"],
            cm_val_norm_path,
            f"{fold_name} - Validation (normalized)",
            normalize="true"
        )

        if eval_out["y_train_score"] is not None:
            save_roc_curves(
                y_train,
                eval_out["y_train_score"],
                roc_train_path,
                f"{fold_name} - Train ROC Curves ({model_name})"
            )

        if eval_out["y_val_score"] is not None:
            save_roc_curves(
                y_val,
                eval_out["y_val_score"],
                roc_val_path,
                f"{fold_name} - Validation ROC Curves ({model_name})"
            )

    metrics_rows.append(eval_out["test_metrics"])
    per_class_parts.append(eval_out["test_per_class"])
    report_blocks.append("CLASSIFICATION REPORT - TEST\n" + "-" * 80 + "\n" + eval_out["test_report"])

    save_confusion_matrix(
        y_test, eval_out["y_test_pred"],
        cm_test_raw_path,
        f"{fold_name} - Test (raw)",
        normalize=None
    )
    save_confusion_matrix(
        y_test, eval_out["y_test_pred"],
        cm_test_norm_path,
        f"{fold_name} - Test (normalized)",
        normalize="true"
    )

    if eval_out["y_test_score"] is not None:
        save_roc_curves(
            y_test,
            eval_out["y_test_score"],
            roc_test_path,
            f"{fold_name} - Test ROC Curves ({model_name})"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    per_class_df = pd.concat(per_class_parts, ignore_index=True)

    metrics_df.to_csv(metrics_csv_path, index=False)
    per_class_df.to_csv(per_class_csv_path, index=False)

    save_metrics_table_png(
        metrics_df,
        metrics_png_path,
        f"Metrics Summary - {model_name} - {fold_name}"
    )

    with open(reports_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Fold: {fold_name}\n")
        f.write(f"File: {npz_path.name}\n")
        f.write(f"Model: {model_name}\n")
        f.write("=" * 100 + "\n\n")

        f.write("Shapes\n")
        f.write("-" * 80 + "\n")
        f.write(f"X_train: {X_train.shape} | y_train: {y_train.shape}\n")
        f.write(f"X_val:   {X_val.shape} | y_val:   {y_val.shape}\n")
        f.write(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}\n\n")

        f.write("Best parameters\n")
        f.write("-" * 80 + "\n")
        f.write(json.dumps(best_params, indent=2, ensure_ascii=False, default=str))
        f.write("\n\n")

        f.write("Best validation metrics\n")
        f.write("-" * 80 + "\n")
        f.write(json.dumps({
            "accuracy_val": float(best_row["accuracy_val"]),
            "f1_weighted_val": float(best_row["f1_weighted_val"]),
            "f1_macro_val": float(best_row["f1_macro_val"]),
            "balanced_acc_val": float(best_row["balanced_acc_val"]),
            "log_loss_val": None if pd.isna(best_row["log_loss_val"]) else float(best_row["log_loss_val"]),
        }, indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Metrics summary\n")
        f.write("-" * 80 + "\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("Per-class metrics\n")
        f.write("-" * 80 + "\n")
        f.write(per_class_df.to_string(index=False))
        f.write("\n\n")

        for block in report_blocks:
            f.write(block)
            f.write("\n\n")

    joblib.dump(eval_out["final_model"], final_model_path)
    joblib.dump(eval_out["best_model"], best_model_path)

    # Fold-level summary row
    fold_summary = {
        "fold_name": fold_name,
        "file_name": npz_path.name,
        "model_name": model_name,
        "best_accuracy_val": float(best_row["accuracy_val"]),
        "best_f1_weighted_val": float(best_row["f1_weighted_val"]),
        "best_f1_macro_val": float(best_row["f1_macro_val"]),
        "best_balanced_acc_val": float(best_row["balanced_acc_val"]),
        "best_log_loss_val": None if pd.isna(best_row["log_loss_val"]) else float(best_row["log_loss_val"]),
        "test_accuracy": float(eval_out["test_metrics"]["accuracy"]),
        "test_precision_weighted": float(eval_out["test_metrics"]["precision_weighted"]),
        "test_recall_weighted": float(eval_out["test_metrics"]["recall_weighted"]),
        "test_f1_weighted": float(eval_out["test_metrics"]["f1_weighted"]),
        "test_precision_macro": float(eval_out["test_metrics"]["precision_macro"]),
        "test_recall_macro": float(eval_out["test_metrics"]["recall_macro"]),
        "test_f1_macro": float(eval_out["test_metrics"]["f1_macro"]),
        "test_balanced_accuracy": float(eval_out["test_metrics"]["balanced_accuracy"]),
        "test_roc_auc_macro_ovr": None if pd.isna(eval_out["test_metrics"]["roc_auc_macro_ovr"]) else float(eval_out["test_metrics"]["roc_auc_macro_ovr"]),
        "output_dir": str(fold_output_dir)
    }

    print("\nSaved fold results to:", fold_output_dir)

    return {
        "fold_summary": fold_summary,
        "test_per_class_df": eval_out["test_per_class"].assign(fold_name=fold_name),
        "y_test_true": eval_out["y_test_true"],
        "y_test_pred": eval_out["y_test_pred"],
        "y_test_score": eval_out["y_test_score"]
    }


# ============================================================
# CROSS-VALIDATION AGGREGATION
# ============================================================

def aggregate_cv_results(fold_outputs, output_root, model_name):
    model_output_dir = Path(output_root) / model_name
    ensure_dir(model_output_dir)

    # Fold summaries
    fold_summary_df = pd.DataFrame([fo["fold_summary"] for fo in fold_outputs])
    fold_summary_csv = model_output_dir / "cross_validation_fold_results.csv"
    fold_summary_df.to_csv(fold_summary_csv, index=False)

    # Aggregate overall metrics across folds
    metric_cols = [
        "test_accuracy",
        "test_precision_weighted",
        "test_recall_weighted",
        "test_f1_weighted",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "test_balanced_accuracy",
        "test_roc_auc_macro_ovr"
    ]

    agg_rows = []
    for col in metric_cols:
        values = fold_summary_df[col].astype(float).values
        agg_rows.append({
            "metric": col,
            "mean": np.mean(values),
            "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
            "min": np.min(values),
            "max": np.max(values),
            "n_folds": len(values)
        })

    aggregate_metrics_df = pd.DataFrame(agg_rows)
    aggregate_metrics_csv = model_output_dir / "cross_validation_aggregate_metrics.csv"
    aggregate_metrics_df.to_csv(aggregate_metrics_csv, index=False)

    # Per-class summary across folds
    per_class_all = pd.concat([fo["test_per_class_df"] for fo in fold_outputs], ignore_index=True)

    per_class_summary = (
        per_class_all
        .groupby("class_name")[["precision", "recall", "f1_score", "support"]]
        .agg(["mean", "std", "min", "max"])
    )
    per_class_summary.columns = ["_".join(col).strip() for col in per_class_summary.columns.values]
    per_class_summary = per_class_summary.reset_index()

    per_class_all_csv = model_output_dir / "cross_validation_test_per_class_all_folds.csv"
    per_class_summary_csv = model_output_dir / "cross_validation_test_per_class_summary.csv"

    per_class_all.to_csv(per_class_all_csv, index=False)
    per_class_summary.to_csv(per_class_summary_csv, index=False)

    # Pooled test predictions across folds
    y_test_all = np.concatenate([fo["y_test_true"] for fo in fold_outputs], axis=0)
    y_pred_all = np.concatenate([fo["y_test_pred"] for fo in fold_outputs], axis=0)

    pooled_report_txt = classification_report(
        y_test_all,
        y_pred_all,
        labels=LABEL_ORDER,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    pooled_report_file = model_output_dir / "pooled_test_classification_report.txt"
    with open(pooled_report_file, "w", encoding="utf-8") as f:
        f.write("Pooled test classification report across CV folds\n")
        f.write("=" * 100 + "\n\n")
        f.write(pooled_report_txt)
        f.write("\n")

    pooled_cm_raw = model_output_dir / "pooled_test_confusion_matrix_raw.svg"
    pooled_cm_norm = model_output_dir / "pooled_test_confusion_matrix_normalized.svg"

    save_confusion_matrix(
        y_test_all, y_pred_all,
        pooled_cm_raw,
        f"Pooled Test Confusion Matrix - {model_name} (raw)",
        normalize=None
    )
    save_confusion_matrix(
        y_test_all, y_pred_all,
        pooled_cm_norm,
        f"Pooled Test Confusion Matrix - {model_name} (normalized)",
        normalize="true"
    )

    # Pooled ROC across folds
    y_score_all = None
    if all(fo["y_test_score"] is not None for fo in fold_outputs):
        y_score_all = np.concatenate([fo["y_test_score"] for fo in fold_outputs], axis=0)

    pooled_roc_path = model_output_dir / "pooled_test_roc_ovr.svg"
    pooled_auc_macro = np.nan
    if y_score_all is not None:
        pooled_auc_macro = save_roc_curves(
            y_test_all,
            y_score_all,
            pooled_roc_path,
            f"Pooled Test ROC Curves - {model_name}"
        )

    # Compact text summary
    summary_txt = model_output_dir / "cross_validation_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Cross-validation summary for model: {model_name}\n")
        f.write("=" * 100 + "\n\n")
        f.write("Aggregate metrics across folds\n")
        f.write("-" * 80 + "\n")
        f.write(aggregate_metrics_df.to_string(index=False))
        f.write("\n\n")
        f.write("Per-fold test metrics\n")
        f.write("-" * 80 + "\n")
        f.write(fold_summary_df.to_string(index=False))
        f.write("\n\n")
        f.write("Per-class summary across folds\n")
        f.write("-" * 80 + "\n")
        f.write(per_class_summary.to_string(index=False))
        f.write("\n\n")
        f.write("Pooled test classification report\n")
        f.write("-" * 80 + "\n")
        f.write(pooled_report_txt)
        f.write("\n\n")

        if not pd.isna(pooled_auc_macro):
            f.write("Pooled ROC-AUC macro OvR\n")
            f.write("-" * 80 + "\n")
            f.write(f"{pooled_auc_macro:.6f}\n")

    print("\n" + "=" * 100)
    print(f"Cross-validation finished for model: {model_name}")
    print("Saved:")
    print(fold_summary_csv)
    print(aggregate_metrics_csv)
    print(per_class_all_csv)
    print(per_class_summary_csv)
    print(pooled_report_file)
    print(pooled_cm_raw)
    print(pooled_cm_norm)
    if y_score_all is not None:
        print(pooled_roc_path)
    print(summary_txt)
    print("=" * 100)

    return {
        "fold_summary_df": fold_summary_df,
        "aggregate_metrics_df": aggregate_metrics_df,
        "per_class_summary_df": per_class_summary
    }


# ============================================================
# MAIN RUNNER
# ============================================================

def run_cross_validation_benchmark(input_dir, output_root, model_name, random_state=42):
    input_dir = Path(input_dir)
    npz_files = sorted(input_dir.glob("*.npz"))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    ensure_dir(Path(output_root) / model_name)

    print("\nFound folds:")
    for f in npz_files:
        print(" -", f.name)

    fold_outputs = []
    for npz_path in npz_files:
        fold_out = process_single_fold(
            npz_path=npz_path,
            output_root=output_root,
            model_name=model_name,
            param_grid=PARAM_GRIDS[model_name],
            random_state=random_state
        )
        fold_outputs.append(fold_out)

    return aggregate_cv_results(
        fold_outputs=fold_outputs,
        output_root=output_root,
        model_name=model_name
    )


if __name__ == "__main__":
    for model_name in MODEL_NAMES:
        run_cross_validation_benchmark(
            input_dir=INPUT_DIR,
            output_root=OUTPUT_ROOT,
            model_name=model_name,
            random_state=RANDOM_STATE
        )