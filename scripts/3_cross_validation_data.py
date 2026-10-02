#!/usr/bin/env python3
"""Assemble all CV conditions without drawing any new split."""

import argparse
from pathlib import Path
import numpy as np
from pipeline_common import (
    load_manifest,
    class_mapping,
    selected_folds,
    read_json,
    write_json,
    sha256,
    require_new,
    check_features,
)


def run(a):
    table, meta = load_manifest(a.data)
    provenance = read_json(a.features / "provenance.json")
    if provenance["manifest_sha256"] != sha256(a.data / "manifest.json"):
        raise ValueError("Feature manifest mismatch")
    if class_mapping(provenance) != class_mapping(meta):
        raise ValueError("Feature and preprocessing class-name metadata differ")
    out = require_new(a.output)
    for family in provenance["families"]:
        raw = np.load(a.features / family / "original.npz", allow_pickle=False)
        ids = raw["sample_ids"].astype(str)
        if len(set(ids)) != len(ids) or set(ids) != set(table.sample_id):
            raise ValueError("Original cache identity mismatch")
        order = {sid: i for i, sid in enumerate(ids)}
        lookup = table.set_index("sample_id")
        if not np.array_equal(raw["labels"], lookup.loc[ids, "label"].to_numpy()):
            raise ValueError("Original label mismatch")
        check_features(raw["embeddings"], raw["feature_names"])
        for fold in selected_folds(meta, a.folds):
            dest = out / family / fold["name"]
            dest.mkdir(parents=True)
            common = {}
            for split, key in [
                ("validation", "val"),
                ("test", "test"),
                ("train", "train_original"),
            ]:
                idx = np.array([order[s] for s in fold[split]])
                for name, source in [
                    ("X", "embeddings"),
                    ("y", "labels"),
                    ("ids", "sample_ids"),
                    ("animals", "animals"),
                ]:
                    common[f"{name}_{key}"] = raw[source][idx]
            for level in [0] + provenance["levels"]:
                if level == 0:
                    X = common["X_train_original"]
                    y = common["y_train_original"]
                    trainids = common["ids_train_original"]
                    animals = common["animals_train_original"]
                    rowids = trainids
                else:
                    aug = np.load(
                        a.features / family / fold["name"] / f"aug_{level}.npz",
                        allow_pickle=False,
                    )
                    if not np.array_equal(raw["feature_names"], aug["feature_names"]):
                        raise ValueError("Feature schema changed")
                    X, y, trainids, animals, rowids = [
                        aug[k]
                        for k in [
                            "embeddings",
                            "labels",
                            "sample_ids",
                            "animals",
                            "row_ids",
                        ]
                    ]
                    if not set(trainids) <= set(fold["train"]):
                        raise ValueError("Augmentation crossed split boundary")
                    if len(set(rowids)) != len(rowids):
                        raise ValueError("Duplicate augmented row IDs")
                    if not np.array_equal(y, lookup.loc[trainids, "label"].to_numpy()):
                        raise ValueError("Augmented labels changed")
                    for animal in set(animals):
                        for label in range(1, 6):
                            if np.sum((animals == animal) & (y == label)) != level:
                                raise ValueError(
                                    "Class/animal sampling target violated"
                                )
                check_features(X, raw["feature_names"])
                np.savez_compressed(
                    dest / f"level_{level}.npz",
                    X_train=X,
                    y_train=y,
                    ids_train=trainids,
                    animals_train=animals,
                    row_ids_train=rowids,
                    feature_names=raw["feature_names"],
                    **common,
                )
    write_json(
        out / "provenance.json",
        dict(
            **provenance,
            features_provenance_sha256=sha256(a.features / "provenance.json"),
            conditions=[0] + provenance["levels"],
        ),
    )
    print(
        "All requested CV conditions assembled; level_0 is the original-only baseline."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["data", "features", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--folds", nargs="+")
    run(p.parse_args())


if __name__ == "__main__":
    main()
