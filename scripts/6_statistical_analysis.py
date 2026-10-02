#!/usr/bin/env python3
"""Descriptive held-out animal comparisons of locked, validation-selected procedures."""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon, PermutationMethod
from pipeline_common import read_json, write_json, sha256, require_new, class_mapping
from evaluation import export_evaluation

METRICS = ["f1_macro", "f1_weighted", "balanced_accuracy"]


def holm(pvalues):
    p = np.array(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    )
    return adjusted


def paired_tests(frame, columns):
    rows = []
    for metric in columns:
        pivot = frame.pivot(index="fold", columns="experiment", values=metric)
        if pivot.isna().any().any():
            raise ValueError("Incomplete pairs; no silent fold deletion")
        for left, right in itertools.combinations(pivot.columns, 2):
            d = (pivot[left] - pivot[right]).to_numpy()
            sd = d.std(ddof=1)
            p = (
                1.0
                if np.all(d == 0)
                else float(
                    wilcoxon(
                        d,
                        alternative="two-sided",
                        method=PermutationMethod(n_resamples=np.inf),
                    ).pvalue
                )
            )
            rows.append(
                dict(
                    metric=metric,
                    left=left,
                    right=right,
                    n_animals=len(d),
                    mean_difference=d.mean(),
                    paired_dz=d.mean() / sd if sd > 0 else np.nan,
                    p_raw=p,
                )
            )
    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm"] = holm(result.p_raw)
        result["reject_holm"] = result.p_holm < 0.05
    return result


def friedman_permutation(frame, seed=42, nperm=10000):
    """Within-animal label permutation of Friedman rank statistic.

    Conditional exploratory test; does not make overlapping CV fits independent.
    """
    pivot = frame.pivot(index="fold", columns="experiment", values="f1_macro")
    if pivot.isna().any().any():
        raise ValueError("Incomplete Friedman matrix")
    n, k = pivot.shape
    if k < 3:
        return {"status": "not_applicable_fewer_than_three_procedures"}
    ranks = np.array([rankdata(-r) for r in pivot.to_numpy()])
    stat = lambda r: float(np.sum((r.sum(axis=0) - n * (k + 1) / 2) ** 2))
    observed = stat(ranks)
    total = __import__("math").factorial(k) ** n
    if total <= 100000:
        perms = list(itertools.permutations(range(k)))
        ge = 0
        for choices in itertools.product(perms, repeat=n):
            shuffled = np.array([r[list(p)] for r, p in zip(ranks, choices)])
            ge += stat(shuffled) >= observed - 1e-12
        p = ge / total
        method = "exact within-animal permutation"
        draws = total
    else:
        rng = np.random.default_rng(seed)
        ge = 0
        for _ in range(nperm):
            ge += (
                stat(np.array([rng.permutation(r) for r in ranks])) >= observed - 1e-12
            )
        p = (ge + 1) / (nperm + 1)
        method = "Monte Carlo within-animal permutation"
        draws = nperm
    return dict(
        rank_sum_statistic=observed,
        p_value=p,
        method=method,
        permutations=draws,
        n_animals=n,
        n_procedures=k,
    )


def plot_boxes(frame, metric, path, title, paired=None):
    import matplotlib.pyplot as plt

    order = sorted(frame.experiment.unique())
    fig, ax = plt.subplots(figsize=(max(7, len(order) * 0.5), 5))
    vals = [frame.loc[frame.experiment == e, metric].to_numpy() for e in order]
    ax.boxplot(
        vals, patch_artist=True, boxprops={"facecolor": "#cfe8ff"}, showfliers=False
    )
    for i, v in enumerate(vals, 1):
        ax.scatter(
            np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)),
            v,
            s=18,
            color="#174a75",
            zorder=3,
        )
    ax.set(
        xticks=np.arange(1, len(order) + 1),
        xticklabels=order,
        ylabel=metric,
        title=title,
        ylim=(0, 1.02),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    if paired is not None and not paired.empty:
        comparisons = paired.loc[paired.metric == metric]
        positions = {name: i for i, name in enumerate(order, 1)}
        # Use a separate margin above the 0..1 score axis, preserving its scale.
        transform = ax.get_xaxis_transform()
        for level, row in enumerate(comparisons.itertuples()):
            left, right = sorted([positions[row.left], positions[row.right]])
            height = 1.03 + level * 0.10
            ax.plot(
                [left, left, right, right],
                [height, height + 0.025, height + 0.025, height],
                transform=transform,
                color="black",
                linewidth=1,
                clip_on=False,
            )
            value = f"{row.p_holm:.3g}" if row.p_holm >= 0.001 else f"{row.p_holm:.2e}"
            ax.text(
                (left + right) / 2,
                height + 0.03,
                f"Holm p = {value}",
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=9,
                clip_on=False,
            )
        ax.set_title("")
        ax.text(
            0.5,
            1.13 + 0.10 * len(comparisons),
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            clip_on=False,
        )
        fig.set_size_inches(max(7, len(order) * 0.5), 5 + 0.45 * len(comparisons))
        fig.tight_layout()
    else:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "none"
    out = require_new(a.output)
    parts = []
    provenance = []
    selected_predictions = []
    valrows = []
    names = None
    for root in a.results:
        ev = read_json(root / "evaluation_provenance.json")
        lock = read_json(root / "selection.json")
        current_names = class_mapping(lock)
        if names is not None and current_names != names:
            raise ValueError("Model families have conflicting class-name metadata")
        names = current_names
        if ev["selection_sha256"] != sha256(root / "selection.json"):
            raise ValueError("Selection changed after evaluation")
        if ev["smoke_test"] and not a.allow_smoke:
            raise ValueError("Refusing to publish smoke-test results")
        provenance.append(dict(root=str(root), selection_sha256=ev["selection_sha256"]))
        frame = pd.read_csv(root / "test_metrics_all_conditions.csv")
        parts.append(frame)
        for r in lock["selection"]:
            family = r.get("family", lock.get("model"))
            valrows.append(
                dict(
                    family=family,
                    fold=r["fold"],
                    model=r.get("model", family),
                    level=r.get("level", "direct"),
                    **r["metrics"],
                )
            )
            chosen = r.get("selected", True)
            if chosen:
                directory = (
                    (root / r["artifact"]).parent / "test"
                    if "family" in r
                    else root / r["fold"] / "test"
                )
                predictions = pd.read_csv(
                    directory / "predictions.csv",
                    dtype={"sample_id": str, "animal": str},
                )
                predictions["family"] = family
                predictions["fold"] = r["fold"]
                selected_predictions.append(predictions)
        for histfile in root.rglob("*history.csv"):
            if histfile.stat().st_size < 5:
                continue
            h = pd.read_csv(histfile)
            if "epoch" not in h:
                continue
            fig, ax = plt.subplots()
            for col in ["train_loss", "train_weighted_loss", "f1_macro"]:
                if col in h:
                    ax.plot(h.epoch, h[col], label=col)
            ax.set(xlabel="Epoch", title=str(histfile.relative_to(root)))
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                out
                / (
                    root.name
                    + "__"
                    + "__".join(histfile.relative_to(root).parts)
                    + ".svg"
                )
            )
            plt.close(fig)
    allrows = pd.concat(parts, ignore_index=True)
    if allrows.duplicated(["family", "fold", "model", "level"]).any():
        raise ValueError("Duplicate experiment/fold rows")
    for _, group in allrows.groupby(["family", "model", "level"]):
        if len(group) != 5 and not a.allow_smoke:
            raise ValueError("Expected exactly five animal folds")
    allrows["experiment"] = (
        allrows.family + " / " + allrows.model + " / " + allrows.level.astype(str)
    )
    allrows.to_csv(out / "test_metrics_by_fold_all_experiments.csv", index=False)
    numeric = [
        c
        for c in allrows
        if c
        in [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "precision_weighted",
            "recall_weighted",
            "roc_auc_macro_ovr",
        ]
        + METRICS
    ]
    allrows.groupby("experiment")[numeric].agg(
        ["mean", "std", "median", "min", "max", "count"]
    ).to_csv(out / "all_conditions_summary.csv")
    val = pd.DataFrame(valrows)
    val.to_csv(out / "validation_selection_metrics.csv", index=False)
    selected = allrows[allrows.selected == True].copy()
    if selected.duplicated(["family", "fold"]).any():
        raise ValueError("Multiple selected models in family/fold")
    selected["experiment"] = selected.family
    selected.to_csv(out / "selected_procedures_by_fold.csv", index=False)
    selected.groupby("family")[numeric].agg(
        ["mean", "std", "median", "min", "max", "count"]
    ).to_csv(out / "selected_procedures_summary.csv")
    paired = paired_tests(selected, METRICS)
    for metric in METRICS:
        plot_boxes(
            selected,
            metric,
            out / f"selected_{metric}.svg",
            "Validation-selected procedures; held-out animals",
            paired=paired,
        )
        for family, group in allrows.groupby("family"):
            plot_boxes(
                group,
                metric,
                out / f"{family}_all_conditions_{metric}.svg",
                "Fixed-condition test results; descriptive comparison",
            )
    paired.to_csv(out / "selected_pairwise_wilcoxon_holm_dz.csv", index=False)
    write_json(
        out / "friedman_permutation.json", friedman_permutation(selected, a.seed)
    )
    ranks = selected.pivot(index="fold", columns="experiment", values="f1_macro").rank(
        axis=1, ascending=False
    )
    ranks.to_csv(out / "animal_ranks.csv")
    ranks.mean().to_csv(out / "average_ranks_descriptive.csv")
    alltests = paired_tests(allrows, METRICS) if a.all_pairwise else pd.DataFrame()
    alltests.to_csv(out / "all_conditions_pairwise_exploratory.csv", index=False)
    predictions = pd.concat(selected_predictions, ignore_index=True)
    for family, group in predictions.groupby("family"):
        if group.sample_id.duplicated().any():
            raise ValueError("Nucleus evaluated in multiple test folds")
        export_evaluation(
            out / f"pooled_{family}",
            group.true_label.to_numpy(),
            group.predicted_label.to_numpy(),
            group[[f"p_{c}" for c in range(1, 6)]].to_numpy(),
            group.sample_id.to_numpy(),
            group.animal.to_numpy(),
            family + " pooled out-of-fold",
            class_names=names,
        )
    sets = predictions.groupby("family").sample_id.apply(set)
    if len(sets) > 1 and not all(s == sets.iloc[0] for s in sets):
        raise ValueError("Families evaluated different nuclei")
    with pd.ExcelWriter(out / "statistical_tables.xlsx", engine="openpyxl") as writer:
        selected.to_excel(writer, sheet_name="Selected by fold", index=False)
        paired.to_excel(writer, sheet_name="Paired tests Holm", index=False)
        allrows.to_excel(writer, sheet_name="All conditions", index=False)
        val.to_excel(writer, sheet_name="Validation selection", index=False)
        if len(alltests):
            alltests.to_excel(
                writer, sheet_name="All pairwise exploratory", index=False
            )
    write_json(
        out / "provenance.json",
        dict(inputs=provenance, seed=a.seed, smoke_test=a.allow_smoke, class_names=names),
    )
    (out / "INTERPRETATION.txt").write_text(
        "Primary unit: animal. Five-class macro-F1 includes absent classes as zero.\n"
        "Models and augmentation levels selected within each fold on clean validation.\n"
        "Pooled metrics weight nuclei; they differ from equal-weight animal means.\n"
        "All-condition comparisons are exploratory and do not define a winner.\n"
        "Holm correction covers all pairs across the three metrics in each exported comparison family.\n"
        "Overlapping CV training sets limit independence; permutation p-values remain exploratory.\n"
        "With five nonzero differences, minimum exact two-sided Wilcoxon p is 0.0625.\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-pairwise", action="store_true")
    p.add_argument("--allow-smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
