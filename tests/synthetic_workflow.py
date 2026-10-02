#!/usr/bin/env python3
"""Reproducible integration smoke test. Synthetic vectors are NOT DINO inference."""

import argparse
import subprocess
import sys
import importlib
from pathlib import Path
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_common import read_json, write_json


def command(*args):
    print("RUN", *map(str, args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def run(out, cnn=False):
    out.mkdir(parents=True, exist_ok=False)
    for kind in ["image", "labels", "class"]:
        (out / kind).mkdir()
    rng = np.random.default_rng(45)
    for animal in range(1, 6):
        shape = (24, 90, 170)
        image = rng.uniform(0.01, 0.02, shape).astype(np.float32)
        labels = np.zeros(shape, np.uint16)
        classes = np.zeros(shape, np.uint8)
        z, y, x = np.indices(shape)
        inst = 0
        for c in range(1, 6):
            for j in range(10):
                inst += 1
                mask = ((z - 12) / (3 + 0.1 * c)) ** 2 + (
                    (y - (9 + 16 * (c - 1))) / (3 + 0.1 * j)
                ) ** 2 + ((x - (9 + 16 * j)) / 3) ** 2 <= 1
                labels[mask] = inst
                classes[mask] = c
                image[mask] = rng.uniform(0.2, 0.8, mask.sum())
        for kind, array in [("image", image), ("labels", labels), ("class", classes)]:
            tifffile.imwrite(out / kind / f"{animal}.tif", array)
    scripts = ROOT / "scripts"
    command(
        scripts / "1_preprocessing.py",
        "--class-names",
        ROOT / "config/class_names.json",
        "--images",
        out / "image",
        "--instances",
        out / "labels",
        "--classes",
        out / "class",
        "--output",
        out / "data",
        "--min-voxels",
        20,
    )
    command(
        scripts / "2_embedding_extraction.py",
        "--data",
        out / "data",
        "--output",
        out / "features",
        "--families",
        "handcrafted",
        "--levels",
        2,
        "--workers",
        0,
        "--initial",
        16,
        16,
        16,
        "--target",
        24,
        24,
        24,
    )
    command(
        scripts / "3_cross_validation_data.py",
        "--data",
        out / "data",
        "--features",
        out / "features",
        "--output",
        out / "cv",
    )
    command(
        scripts / "4_run_models.py",
        "--phase",
        "select",
        "--cv",
        out / "cv",
        "--output",
        out / "classical",
        "--smoke",
    )
    command(
        scripts / "4_run_models.py",
        "--phase",
        "evaluate",
        "--cv",
        out / "cv",
        "--output",
        out / "classical",
    )
    # Figure logic test with synthetic 1024D vectors and exact original matching.
    features = out / "features"
    (features / "dino").mkdir()
    hand = np.load(features / "handcrafted/original.npz")
    D = rng.normal(size=(len(hand["labels"]), 1024)).astype(np.float32)
    np.savez_compressed(
        features / "dino/original.npz",
        embeddings=D,
        labels=hand["labels"],
        sample_ids=hand["sample_ids"],
        animals=hand["animals"],
        feature_names=np.array([f"dino_{i}" for i in range(1024)]),
    )
    command(
        ROOT / "scripts/7_representation_analysis.py",
        "--features",
        features,
        "--output",
        out / "exploration",
    )
    command(
        ROOT / "scripts/6_statistical_analysis.py",
        "--results",
        out / "classical",
        "--output",
        out / "statistics",
        "--allow-smoke",
        "--all-pairwise",
    )
    if cnn:
        command(
            scripts / "5_run_cNN.py",
            "--phase",
            "select",
            "--data",
            out / "data",
            "--output",
            out / "cnn",
            "--smoke",
            "--folds",
            "test_img_1",
            "--workers",
            0,
            "--batch-size",
            10,
            "--device",
            "cpu",
            "--initial",
            32,
            32,
            32,
            "--target",
            32,
            32,
            32,
        )
        command(
            scripts / "5_run_cNN.py",
            "--phase",
            "evaluate",
            "--data",
            out / "data",
            "--output",
            out / "cnn",
            "--workers",
            0,
            "--device",
            "cpu",
        )
    write_json(
        out / "SYNTHETIC_ONLY.json",
        {
            "biological_results": False,
            "DINO_inference_tested": False,
            "CNN_tested": cnn,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cnn", action="store_true")
    a = p.parse_args()
    run(a.output, a.cnn)
