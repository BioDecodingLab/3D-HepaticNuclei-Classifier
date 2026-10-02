"""Shared contracts, provenance and split validation. No learned preprocessing here."""

from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd

LABELS = [1, 2, 3, 4, 5]
CLASS_NAMES = [f"Class {label}" for label in LABELS]
LEVELS = [100, 200, 500, 1000, 2000, 4000]
SCHEMA = "hepatic-nuclei-v2"


def class_mapping(metadata=None):
    """Validate display metadata; numeric learning targets are unchanged."""
    mapping = (metadata or {}).get("class_names")
    if mapping is None:
        return dict(zip(map(str, LABELS), CLASS_NAMES))
    if not isinstance(mapping, dict) or set(mapping) != set(map(str, LABELS)):
        raise ValueError("class_names must map exactly the string IDs 1 through 5")
    if any(not isinstance(v, str) or not v.strip() for v in mapping.values()):
        raise ValueError("Class names must be nonempty strings")
    mapping = {str(k): mapping[str(k)].strip() for k in LABELS}
    if len(set(mapping.values())) != len(LABELS):
        raise ValueError("Class names must be unique")
    return mapping


def stable_seed(*parts):
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little"
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def default(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)

    tmp.write_text(json.dumps(obj, indent=2, default=default, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def environment():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in [
        "numpy",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "torch",
        "tifffile",
        "umap-learn",
        "pandas",
    ]:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def load_manifest(root):
    root = Path(root)
    meta = read_json(root / "manifest.json")
    class_mapping(meta)
    if meta["schema"] != SCHEMA:
        raise ValueError("Unsupported manifest schema")
    table = pd.read_csv(root / "samples.csv", dtype={"sample_id": str, "animal": str})
    if table.sample_id.duplicated().any():
        raise ValueError("Duplicate original IDs")
    if sha256(root / "samples.csv") != meta["samples_sha256"]:
        raise ValueError("Sample table changed")
    for fold in meta["folds"]:
        validate_split(table, fold)
    return table, meta


def validate_split(table, fold):
    sets = [set(fold[s]) for s in ["train", "validation", "test"]]
    if any(
        len(values) != len(fold[key])
        for values, key in zip(sets, ["train", "validation", "test"])
    ):
        raise ValueError("Repeated identity within a split")
    if any(not s for s in sets):
        raise ValueError("Empty split")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Original nucleus crosses splits")
    if set.union(*sets) != set(table.sample_id):
        raise ValueError("Split coverage mismatch")
    lookup = table.set_index("sample_id")
    test_animals = set(lookup.loc[fold["test"], "animal"])
    if test_animals != {fold["animal"]}:
        raise ValueError("Invalid held-out animal")
    if fold["animal"] in set(lookup.loc[fold["train"] + fold["validation"], "animal"]):
        raise ValueError("Held-out animal enters development")
    for split in ["train", "validation"]:
        if set(lookup.loc[fold[split], "label"]) != set(LABELS):
            raise ValueError(f"{split} must represent all five classes")


def selected_folds(meta, names=None):
    if names and not set(names) <= {f["name"] for f in meta["folds"]}:
        raise ValueError("Unknown fold")
    return [f for f in meta["folds"] if not names or f["name"] in names]


def require_new(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to mix runs in nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_features(X, names=None):
    if X.ndim != 2 or not len(X):
        raise ValueError("Expected nonempty 2D features")
    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X).all(axis=0))[0]
        raise ValueError(
            f"Nonfinite feature columns: {bad.tolist()}; inspect QC before fitting"
        )
    if names is not None and len(names) != X.shape[1]:
        raise ValueError("Feature schema mismatch")


def verify_patch_files(root, table):
    """Detect stale/replaced patches before expensive extraction or direct fitting."""
    for row in table.to_dict("records"):
        for kind in ["image", "mask"]:
            path = Path(root) / row[kind + "_path"]
            if sha256(path) != row[kind + "_sha256"]:
                raise ValueError(f"Changed {kind} for {row['sample_id']}")
