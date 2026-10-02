#!/usr/bin/env python3
"""Extract original intensity/mask patches, inspect sizes and freeze outer-fold splits."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
from pipeline_common import (
    LABELS,
    class_mapping,
    read_json,
    SCHEMA,
    sha256,
    write_json,
    environment,
    require_new,
    validate_split,
)


def normalize_by_percentile(image, pmin=30.0, pmax=99.999):
    image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite intensity")
    lo, hi = np.percentile(image, [pmin, pmax])
    if hi <= lo:
        raise ValueError("Degenerate image normalization")
    return np.clip((image - lo) / (hi - lo + 1e-30), 0, 1), float(lo), float(hi)


def border_ids(labels, margin):
    border = np.zeros(labels.shape, bool)
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, margin), slice(-margin, None)
        border[tuple(lo)] = True
        border[tuple(hi)] = True
    return set(np.unique(labels[border])) - {0}


def discover(folder):
    files = sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in {".tif", ".tiff"}
    )
    result = {p.stem: p for p in files}
    if len(result) != len(files):
        raise ValueError("Duplicate TIFF stems")
    return result


def plot_inspection(table, output, names=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "none"
    out = Path(output)
    names = class_mapping({"class_names": names})
    out.mkdir(parents=True, exist_ok=True)
    measurements = [
        "depth",
        "height",
        "width",
        "bbox_voxels",
        "mask_voxels",
        "original_volume_um3",
    ]
    for col in measurements:
        fig, ax = plt.subplots()
        ax.hist(table[col], bins=30, edgecolor="white")
        ax.set(xlabel=col, ylabel="Original nuclei")
        fig.tight_layout()
        fig.savefig(out / f"{col}_global.svg")
        plt.close(fig)
        fig, ax = plt.subplots()
        for label, group in table.groupby("label"):
            ax.hist(group[col], bins=30, histtype="step", label=names[str(label)])
        ax.set(xlabel=col, ylabel="Original nuclei")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{col}_by_class.svg")
        plt.close(fig)
    table.groupby(["animal", "label"])[measurements].agg(
        ["count", "mean", "std", "min", "max"]
    ).to_csv(out / "size_statistics.csv")
    counts = (
        table.groupby(["animal", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )
    counts.to_csv(out / "animal_class_counts.csv")
    ax = counts.rename(columns={k: names[str(k)] for k in LABELS}).plot.bar(stacked=True, figsize=(8, 5))
    ax.set_ylabel("Original nuclei")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "animal_class_counts.svg")
    plt.close(ax.figure)


def create_splits(table, seed=42):
    """10% of each outer development pool; stratify jointly by animal and class."""
    folds = []
    for animal in sorted(table.animal.unique()):
        pool = table[table.animal != animal]
        strata = pool.animal.astype(str) + ":" + pool.label.astype(str)
        counts = strata.value_counts()
        nval = int(np.ceil(0.1 * len(pool)))
        if counts.min() < 2 or nval < len(counts) or len(pool) - nval < len(counts):
            raise ValueError(
                f"Cannot represent animal/class strata in a 10% split for {animal}; review counts, do not silently change split"
            )
        train, val = train_test_split(
            pool.sample_id.to_numpy(), test_size=0.1, stratify=strata, random_state=seed
        )
        fold = dict(
            name=f"test_{animal}",
            animal=animal,
            train=sorted(train),
            validation=sorted(val),
            test=sorted(table.loc[table.animal == animal, "sample_id"]),
        )
        validate_split(table, fold)
        folds.append(fold)
    return folds


def run(args):
    names_file = getattr(args, "class_names", None)
    names = class_mapping({"class_names": read_json(names_file) if names_file else None})
    out = require_new(args.output)
    images, masks, classes = map(discover, [args.images, args.instances, args.classes])
    if not images or set(images) != set(masks) or set(images) != set(classes):
        raise ValueError("Image, instance and class TIFF stems must match exactly")
    if len(images) != args.expected_animals:
        raise ValueError("Unexpected animal/image count")
    rows, image_qc, object_qc, sources = [], [], [], []
    for name, image_path in images.items():
        print(f"Extracting {name}", flush=True)
        image, instances, classmap = [
            tifffile.imread(p) for p in [image_path, masks[name], classes[name]]
        ]
        if (
            image.ndim != 3
            or image.shape != instances.shape
            or image.shape != classmap.shape
        ):
            raise ValueError("Expected matching single-channel 3D arrays")
        if not np.issubdtype(instances.dtype, np.integer) or instances.min() < 0:
            raise ValueError("Invalid instance labels")
        if not np.issubdtype(classmap.dtype, np.integer) or not set(
            np.unique(classmap)
        ) <= {0, *LABELS}:
            raise ValueError("Class map must contain integers 0..5")
        image, lo, hi = normalize_by_percentile(image)
        excluded = border_ids(instances, args.border_margin)
        animal = "img_" + name
        for kind, path in [
            ("image", image_path),
            ("instances", masks[name]),
            ("classes", classes[name]),
        ]:
            sources.append(
                dict(
                    animal=animal,
                    kind=kind,
                    path=str(path.resolve()),
                    sha256=sha256(path),
                )
            )
        for region in regionprops(instances):
            sid = f"{animal}__nucleus_{region.label}"
            reason = (
                "border"
                if region.label in excluded
                else "small" if region.area < args.min_voxels else ""
            )
            mask = region.image.astype(
                bool
            )  # Exact instance; no threshold or hole filling.
            vals = classmap[region.slice][mask]
            nonzero = vals[vals > 0]
            if not len(nonzero):
                reason = reason or "unclassified"
            if reason:
                object_qc.append(dict(sample_id=sid, status=reason))
                continue
            counts = np.bincount(nonzero, minlength=6)
            label = int(
                counts[1:].argmax() + 1
            )  # Original majority rule; ties -> smallest label.
            patch = image[region.slice].copy()
            patch[~mask] = 0
            rel = Path("patches") / animal / f"label_{label}" / (sid + ".tif")
            maskrel = Path("masks") / animal / f"label_{label}" / (sid + ".tif")
            for dest in [out / rel, out / maskrel]:
                dest.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                out / rel, patch, metadata={"axes": "ZYX", "spacing": 0.3, "unit": "um"}
            )
            tifffile.imwrite(
                out / maskrel,
                mask.astype(np.uint8),
                metadata={"axes": "ZYX", "spacing": 0.3, "unit": "um"},
            )
            rows.append(
                dict(
                    sample_id=sid,
                    animal=animal,
                    original_instance=int(region.label),
                    label=label,
                    image_path=str(rel),
                    mask_path=str(maskrel),
                    depth=patch.shape[0],
                    height=patch.shape[1],
                    width=patch.shape[2],
                    bbox_voxels=patch.size,
                    mask_voxels=int(mask.sum()),
                    original_volume_um3=float(mask.sum() * 0.3**3),
                    image_sha256=sha256(out / rel),
                    mask_sha256=sha256(out / maskrel),
                )
            )
            object_qc.append(
                dict(
                    sample_id=sid,
                    status="saved",
                    label=label,
                    labeled_fraction=float(len(nonzero) / mask.sum()),
                    majority_fraction=float(counts[label] / len(nonzero)),
                    class_tie=bool((counts[1:] == counts[label]).sum() > 1),
                )
            )
        image_qc.append(
            dict(
                animal=animal,
                lower_percentile=lo,
                upper_percentile=hi,
                excluded_border=len(excluded),
            )
        )
    table = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    if table.empty:
        raise ValueError("No valid nuclei")
    table.to_csv(out / "samples.csv", index=False)
    folds = create_splits(table, args.seed)
    meta = dict(
        schema=SCHEMA,
        class_names=names,
        seed=args.seed,
        validation_fraction=0.1,
        spacing_um=[0.3] * 3,
        normalization_percentiles=[30, 99.999],
        min_voxels=args.min_voxels,
        border_margin=args.border_margin,
        samples_sha256=sha256(out / "samples.csv"),
        folds=folds,
        environment=environment(),
        sources=sources,
    )
    write_json(out / "manifest.json", meta)
    pd.DataFrame(image_qc).to_csv(out / "image_qc.csv", index=False)
    pd.DataFrame(object_qc).to_csv(out / "object_qc.csv", index=False)
    splitrows = []
    lookup = table.set_index("sample_id")
    for f in folds:
        for split in ["train", "validation", "test"]:
            for sid in f[split]:
                splitrows.append(
                    dict(
                        fold=f["name"],
                        split=split,
                        sample_id=sid,
                        animal=lookup.loc[sid, "animal"],
                        label=lookup.loc[sid, "label"],
                    )
                )
    pd.DataFrame(splitrows).to_csv(out / "split_assignments.csv", index=False)
    plot_inspection(table, out / "inspection", names)
    print(f"Saved {len(table)} originals and {len(folds)} frozen splits to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["images", "instances", "classes", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-names", type=Path, help="JSON mapping class IDs 1..5 to display names; inherited by all downstream stages")
    p.add_argument("--min-voxels", type=int, default=2500)
    p.add_argument("--border-margin", type=int, default=3)
    p.add_argument("--expected-animals", type=int, default=5)
    a = p.parse_args()
    if a.border_margin < 1 or a.min_voxels < 1:
        p.error("Positive thresholds required")
    run(a)


if __name__ == "__main__":
    main()
