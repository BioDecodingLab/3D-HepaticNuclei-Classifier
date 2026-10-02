"""Fixed five-class metrics and SVG diagnostic panels."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    log_loss,
    classification_report,
)
from pipeline_common import LABELS, class_mapping, write_json


def normalized_probabilities(prob):
    prob = np.asarray(prob, dtype=np.float64)
    if (
        prob.ndim != 2
        or prob.shape[1] != 5
        or not np.isfinite(prob).all()
        or (prob < 0).any()
    ):
        raise ValueError("Invalid class probabilities")
    sums = prob.sum(axis=1, keepdims=True)
    if not np.allclose(sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("Probabilities do not sum to one")
    return prob / sums  # Only floating-point/CSV roundoff; no learned calibration.


def metrics(y, pred, prob=None):
    result = {"accuracy": float(accuracy_score(y, pred))}
    for average in ["macro", "weighted"]:
        pr, re, f, _ = precision_recall_fscore_support(
            y, pred, labels=LABELS, average=average, zero_division=0
        )
        result.update(
            {
                f"precision_{average}": float(pr),
                f"recall_{average}": float(re),
                f"f1_{average}": float(f),
            }
        )
    # Balanced accuracy over represented true classes; explicit supports accompany it.
    recalls = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)[
        1
    ]
    present = np.array([np.any(np.asarray(y) == c) for c in LABELS])
    result["balanced_accuracy"] = float(recalls[present].mean())
    result["n_present_classes"] = int(present.sum())
    if prob is not None:
        prob = normalized_probabilities(prob)
        if prob.shape != (len(y), 5) or not np.isfinite(prob).all():
            raise ValueError("Invalid class probabilities")
        result["log_loss"] = float(log_loss(y, prob, labels=LABELS))
        aucs = [
            (
                roc_auc_score(np.asarray(y) == c, prob[:, i])
                if np.any(np.asarray(y) == c) and np.any(np.asarray(y) != c)
                else np.nan
            )
            for i, c in enumerate(LABELS)
        ]
        # Strict five-class macro AUC undefined when any class is unobservable.
        result["roc_auc_macro_ovr"] = (
            float(np.mean(aucs)) if np.isfinite(aucs).all() else None
        )
        result["roc_auc_present_classes"] = (
            float(np.nanmean(aucs)) if np.isfinite(aucs).any() else None
        )
    return result


def score_tuple(m):
    return (m["f1_macro"], m["balanced_accuracy"], -m["log_loss"])


def export_evaluation(out, y, pred, prob, ids, animals, title="Test", class_names=None):
    mapping = class_mapping({"class_names": class_names})
    names = [mapping[str(k)] for k in LABELS]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white"}
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prob = normalized_probabilities(prob)
    m = metrics(y, pred, prob)
    write_json(out / "metrics.json", m)
    frame = pd.DataFrame(
        {"sample_id": ids, "animal": animals, "true_label": y, "predicted_label": pred}
    )
    frame["true_class"] = [mapping[str(int(k))] for k in y]
    frame["predicted_class"] = [mapping[str(int(k))] for k in pred]
    write_json(out / "class_names.json", mapping)
    for i, c in enumerate(LABELS):
        frame[f"p_{c}"] = prob[:, i]
    frame.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame([m]).to_csv(out / "metrics_summary.csv", index=False)
    report = classification_report(
        y,
        pred,
        labels=LABELS,
        target_names=names,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(out / "per_class_metrics.csv")
    (out / "classification_report.txt").write_text(
        classification_report(
            y, pred, labels=LABELS, target_names=names, zero_division=0, digits=5
        )
    )
    cm = confusion_matrix(y, pred, labels=LABELS)
    pd.DataFrame(cm, index=names, columns=names).to_csv(
        out / "confusion_counts.csv"
    )
    for normalized in [False, True]:
        values = (
            np.divide(
                cm,
                cm.sum(axis=1, keepdims=True),
                out=np.zeros_like(cm, dtype=float),
                where=cm.sum(axis=1, keepdims=True) > 0,
            )
            * 100
            if normalized
            else cm
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=100 if normalized else None)
        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{values[i,j]:.1f}" if normalized else str(values[i, j]),
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > values.max() / 2 else "black",
                )
        ax.set(
            xticks=range(5),
            yticks=range(5),
            xticklabels=names,
            yticklabels=names,
            xlabel="Predicted class",
            ylabel="True class",
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        fig.colorbar(im, ax=ax, label="Row percentage" if normalized else "Nuclei")
        fig.tight_layout()
        fig.savefig(
            out / ("confusion_percent.svg" if normalized else "confusion_counts.svg")
        )
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    aucrows = []
    curverows = []
    for i, c in enumerate(LABELS):
        binary = np.asarray(y) == c
        if binary.all() or not binary.any():
            aucrows.append({"class": names[i], "auc": np.nan})
            continue
        fpr, tpr, threshold = roc_curve(binary, prob[:, i])
        auc = roc_auc_score(binary, prob[:, i])
        ax.plot(fpr, tpr, label=f"{names[i]} ({auc:.3f})")
        aucrows.append({"class": names[i], "auc": auc})
        curverows.extend(
            {"class": names[i], "fpr": f, "tpr": t, "threshold": th}
            for f, t, th in zip(fpr, tpr, threshold)
        )
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(
        xlabel="False positive rate", ylabel="True positive rate", title=title + " ROC"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc.svg")
    plt.close(fig)
    pd.DataFrame(aucrows).to_csv(out / "auc_per_class.csv", index=False)
    pd.DataFrame(curverows).to_csv(out / "roc_coordinates.csv", index=False)
    return m
