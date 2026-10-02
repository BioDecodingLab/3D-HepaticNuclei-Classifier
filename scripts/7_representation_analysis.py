#!/usr/bin/env python3
"""Post hoc original-nucleus PCA/UMAP and Figure 5 correlation panels; no predictive outputs."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import rankdata
from pipeline_common import require_new, read_json, write_json, sha256, class_mapping, LABELS

COLORS = ["#e41a1c", "#2356a4", "#14843b", "#ffa500", "#842593"]


def correlations(A, B, method="pearson"):
    if method == "spearman":
        A = np.apply_along_axis(rankdata, 0, A)
        B = np.apply_along_axis(rankdata, 0, B)
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    denom = np.linalg.norm(Ac, axis=0)[:, None] * np.linalg.norm(Bc, axis=0)[None, :]
    return np.divide(
        Ac.T @ Bc,
        denom,
        out=np.full((A.shape[1], B.shape[1]), np.nan),
        where=denom > 1e-12,
    )


def heatmap(C, rows, cols, path, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 11))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#cccccc")
    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set(
        xticks=range(len(cols)),
        yticks=range(len(rows)),
        xticklabels=cols,
        yticklabels=rows,
        title=title,
        xlabel="Handcrafted descriptor",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.colorbar(im, ax=ax, label="Correlation", shrink=0.75)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def scatter(coords, y, animals, path, xlabel, ylabel, title, class_names=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    mapping = class_mapping({"class_names": class_names})
    names = sorted(set(animals))
    if len(names) > len(markers):
        raise ValueError("Too many animals for marker map")
    fig, ax = plt.subplots(figsize=(9, 6))
    for ai, animal in enumerate(names):
        for label, color in enumerate(COLORS, 1):
            idx = (animals == animal) & (y == label)
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                s=9,
                alpha=0.5,
                c=color,
                marker=markers[ai],
                linewidths=0,
            )
    handles = [
        Line2D([], [], marker="o", linestyle="", color=c, label=n)
        for c, n in zip(COLORS, [mapping[str(k)] for k in LABELS])
    ]
    handles += [
        Line2D([], [], marker=markers[i], linestyle="", color="gray", label=n)
        for i, n in enumerate(names)
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(a):
    class_names = class_mapping(read_json(a.features / "provenance.json"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "none"
    out = require_new(a.output)
    dino = np.load(a.features / "dino/original.npz", allow_pickle=False)
    hand = np.load(a.features / "handcrafted/original.npz", allow_pickle=False)
    ids = dino["sample_ids"]
    other = hand["sample_ids"]
    index = {s: i for i, s in enumerate(other)}
    if len(set(ids)) != len(ids) or len(index) != len(other) or set(ids) != set(other):
        raise ValueError("Need one matching original per nucleus in both families")
    order = np.array([index[s] for s in ids])
    H = hand["embeddings"][order]
    D = dino["embeddings"]
    names = hand["feature_names"]
    y = dino["labels"]
    animals = dino["animals"]
    if not np.array_equal(y, hand["labels"][order]) or not np.array_equal(
        animals, hand["animals"][order]
    ):
        raise ValueError("Matched labels/animals differ")
    if H.shape[1] != 31 or min(D.shape) < 31:
        raise ValueError(
            "Figure 5 needs 31 handcrafted features and at least 31 PCA components"
        )
    if not np.isfinite(H).all() or not np.isfinite(D).all():
        raise ValueError("Nonfinite representations")
    display_names = [class_names[str(int(k))] for k in y]
    pd.DataFrame(dict(sample_id=ids, label=y, class_name=display_names, animal=animals)).to_csv(
        out / "matched_originals.csv", index=False
    )
    fitted = {}
    for family, X in [("handcrafted", H), ("dino", D)]:
        scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(31, *X.shape), svd_solver="full")
        P = pca.fit_transform(scaled)
        fitted[family] = (P, pca)
        coord = pd.DataFrame(P, columns=[f"PC{i+1}" for i in range(P.shape[1])])
        coord.insert(0, "sample_id", ids)
        coord["animal"] = animals
        coord["label"] = y
        coord["class_name"] = display_names
        coord.to_csv(out / f"{family}_pca_coordinates.csv", index=False)
        pd.DataFrame(
            {
                "component": np.arange(1, P.shape[1] + 1),
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }
        ).to_csv(out / f"{family}_pca_variance.csv", index=False)
        pd.DataFrame(
            pca.components_,
            columns=names if family == "handcrafted" else dino["feature_names"],
        ).to_csv(out / f"{family}_pca_loadings.csv", index=False)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            scatter(
                P[:, [i, j]],
                y,
                animals,
                out / f"{family}_PC{i+1}_PC{j+1}.svg",
                f"PC{i+1} ({100*pca.explained_variance_ratio_[i]:.2f}%)",
                f"PC{j+1} ({100*pca.explained_variance_ratio_[j]:.2f}%)",
                family + " original nuclei",
                class_names=class_names,
            )
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(1, P.shape[1] + 1), np.cumsum(pca.explained_variance_ratio_), "o-"
        )
        ax.set(
            xlabel="Principal components",
            ylabel="Cumulative explained variance",
            ylim=(0, 1.02),
        )
        fig.tight_layout()
        fig.savefig(out / f"{family}_pca_scree.svg")
        plt.close(fig)
        if not a.skip_umap:
            import umap

            U = umap.UMAP(
                n_components=2,
                n_neighbors=a.neighbors,
                min_dist=a.min_dist,
                random_state=a.seed,
                n_jobs=1,
            ).fit_transform(scaled)
            scatter(
                U,
                y,
                animals,
                out / f"{family}_UMAP.svg",
                "UMAP 1",
                "UMAP 2",
                family + " original nuclei; exploratory",
                class_names=class_names,
            )
            pd.DataFrame(
                dict(
                    sample_id=ids, animal=animals, label=y, class_name=display_names, UMAP1=U[:, 0], UMAP2=U[:, 1]
                )
            ).to_csv(out / f"{family}_umap_coordinates.csv", index=False)
    P, pca = fitted["dino"]
    variance = D.var(axis=0, ddof=1)
    dimensions = np.argsort(-variance, kind="stable")[
        :31
    ]  # Raw variance, before StandardScaler.
    pd.DataFrame(
        dict(dimension_zero_based=dimensions, variance=variance[dimensions])
    ).to_csv(out / "figure5_high_variance_dimensions.csv", index=False)
    rowsA = [
        f"PC{i+1} ({100*v:.2f}%)"
        for i, v in enumerate(pca.explained_variance_ratio_[:31])
    ]
    rowsB = [f"DINO dimension {i} (0-based)" for i in dimensions]
    for method in ["pearson", "spearman"]:
        for panel, A, rows in [("A", P[:, :31], rowsA), ("B", D[:, dimensions], rowsB)]:
            C = correlations(A, H, method)
            pd.DataFrame(C, index=rows, columns=names).to_csv(
                out / f"figure5_{panel}_{method}.csv"
            )
            heatmap(
                C,
                rows,
                names,
                out / f"figure5_{panel}_{method}.svg",
                f"Figure 5{panel}: {method.capitalize()} correlation; original nuclei",
            )
            for groupname, groups in [("animal", animals), ("class", y)]:
                for value in sorted(set(groups)):
                    keep = groups == value
                    if keep.sum() < 3:
                        continue
                    C = correlations(A[keep], H[keep], method)
                    pd.DataFrame(C, index=rows, columns=names).to_csv(
                        out / f"figure5_{panel}_{method}_within_{groupname}_{value}.csv"
                    )
    write_json(
        out / "provenance.json",
        dict(
            features_provenance_sha256=sha256(a.features / "provenance.json"),
            class_names=class_names,
            dino_cache_sha256=sha256(a.features / "dino/original.npz"),
            handcrafted_cache_sha256=sha256(a.features / "handcrafted/original.npz"),
            n_originals=len(ids),
            seed=a.seed,
            neighbors=a.neighbors,
            min_dist=a.min_dist,
            umap_skipped=a.skip_umap,
            predictive_use=False,
            notes="Post hoc original-only exploration; pooled correlations may reflect class/animal composition. No nucleus-level significance claims.",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neighbors", type=int, default=50)
    p.add_argument("--min-dist", type=float, default=0.5)
    p.add_argument(
        "--skip-umap", action="store_true", help="Explicit PCA/correlation-only run"
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
