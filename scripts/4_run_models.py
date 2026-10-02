#!/usr/bin/env python3
"""Select on clean validation, then evaluate immutable saved models in a separate phase."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pipeline_common import (
    read_json,
    write_json,
    sha256,
    class_mapping,
)
from model_grids import PARAM_GRIDS
from evaluation import export_evaluation

from parallel_selection import select


def evaluate(a):
    lock = read_json(a.output / "selection.json")
    if sha256(a.cv / "provenance.json") != lock["cv_provenance_sha256"]:
        raise ValueError("CV provenance changed")
    rows = []
    # Every condition is already locked; all-condition outputs are descriptive only.
    for r in lock["selection"]:
        file = a.cv / r["cv_file"]
        modelpath = a.output / r["artifact"]
        if sha256(file) != r["cv_sha256"] or sha256(modelpath) != r["model_sha256"]:
            raise ValueError("Locked artifact changed")
        data = np.load(file, allow_pickle=False)
        model = joblib.load(modelpath)
        for split in ["val", "test"]:
            y = data["y_" + split]
            X = data["X_" + split]
            m = export_evaluation(
                modelpath.parent / split,
                y,
                model.predict(X),
                model.predict_proba(X),
                data["ids_" + split],
                data["animals_" + split],
                f"{r['family']} {r['fold']} {split}",
                class_names=class_mapping(lock),
            )
            if split == "test":
                rows.append(
                    dict(
                        family=r["family"],
                        fold=r["fold"],
                        model=r["model"],
                        level=r["level"],
                        selected=r["selected"],
                        **m,
                    )
                )
    pd.DataFrame(rows).to_csv(a.output / "test_metrics_all_conditions.csv", index=False)
    write_json(
        a.output / "evaluation_provenance.json",
        {
            "selection_sha256": sha256(a.output / "selection.json"),
            "smoke_test": lock["smoke_test"],
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["select", "evaluate"], required=True)
    p.add_argument("--cv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--families", nargs="+", choices=["dino", "handcrafted"])
    p.add_argument(
        "--models", nargs="+", choices=list(PARAM_GRIDS), default=list(PARAM_GRIDS)
    )
    p.add_argument("--levels", type=int, nargs="+")
    p.add_argument("--folds", nargs="+")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny verification run, not publication results",
    )
    p.add_argument("--jobs", type=int, default=1, help="Concurrent condition searches on this host")
    p.add_argument("--threads-per-job", type=int, default=1, help="BLAS/OpenMP threads per worker")
    p.add_argument("--resume", action="store_true", help="Resume checkpoints written by this version")
    p.add_argument("--cache-dir", type=Path, help="Fast local disk for preprocessing caches")
    p.add_argument("--no-cache", action="store_true", help="Disable preprocessing cache")
    a = p.parse_args()
    (select if a.phase == "select" else evaluate)(a)


if __name__ == "__main__":
    main()
