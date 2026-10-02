#!/usr/bin/env python3
"""Original and training-only augmented representations for every requested level."""

import argparse
from pathlib import Path
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pipeline_common import (
    LEVELS,
    LABELS,
    load_manifest,
    selected_folds,
    stable_seed,
    write_json,
    sha256,
    environment,
    require_new,
    verify_patch_files,
)
from dataset_helper import Tif3DDatasetSingle, load_pair, augment_pair
from nuclear_features import extract_features, FEATURE_NAMES


def make_plan(table, fold, max_level, seed):
    pool = table[table.sample_id.isin(fold["train"])]
    rows = []
    for animal in sorted(pool.animal.unique()):
        for label in LABELS:
            records = (
                pool[(pool.animal == animal) & (pool.label == label)]
                .sort_values("sample_id")
                .to_dict("records")
            )
            if not records:
                raise ValueError(f"Missing training class {label} in {animal}")
            rng = np.random.default_rng(
                stable_seed(seed, fold["name"], animal, label, "sampling")
            )
            for draw in range(max_level):
                row = dict(records[int(rng.integers(len(records)))])
                row.update(
                    draw=draw,
                    augmentation_seed=stable_seed(
                        seed, fold["name"], animal, label, draw, "augmentation"
                    ),
                    row_id=f"{fold['name']}__{animal}__class{label}__draw{draw}",
                )
                rows.append(row)
    return rows


def feature_task(task):
    root, row, initial, target, aug = task
    # Keep each process single-threaded when many extraction workers are used.
    torch.set_num_threads(1)
    image, mask = load_pair(root, row, initial, target)
    if aug:
        image, mask = augment_pair(image, mask, int(row["augmentation_seed"]))
    try:
        return extract_features(image, mask)
    except Exception as e:
        raise RuntimeError(f"Feature failure for {row['sample_id']}: {e}") from e


def validate_dino_paths(repo, config, weights):
    if not repo or not config or not weights:
        raise ValueError("DINO requires --dino-repo --dino-config --dino-weights")
    repo = Path(repo).resolve()
    if not (repo / "dinov2").is_dir():
        raise FileNotFoundError(f"No dinov2 source package in {repo}")
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"DINO weights do not exist: {weights}")
    config = Path(config)
    # The upstream function appends .yaml and accepts a configuration stem.
    if config.suffix != ".yaml":
        config = Path(str(config) + ".yaml")
    if not config.is_absolute() and not config.is_file():
        config = repo / "dinov2/configs" / config
    config = config.resolve()
    if not config.is_file():
        raise FileNotFoundError(f"DINO configuration does not exist: {config}")
    return repo, config, weights


def load_dino(repo, config, weights, device):
    repo, config, weights = validate_dino_paths(repo, config, weights)
    if torch.device(device).type != "cuda":
        raise ValueError("The supported upstream 3DINO loader requires CUDA")
    cuda_device = torch.device(device)
    torch.cuda.set_device(cuda_device.index if cuda_device.index is not None else 0)
    sys.path.insert(0, str(repo))
    from dinov2.eval.setup import build_model_for_eval
    from dinov2.configs import load_and_merge_config_3d

    model = build_model_for_eval(
        load_and_merge_config_3d(str(config.with_suffix(""))), str(weights)
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract(root, records, args, family, model=None, aug=False):
    if family == "handcrafted":
        tasks = (
            (root, row, tuple(args.initial), tuple(args.target), aug) for row in records
        )
        if args.workers:
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context("spawn")
            ) as ex:
                results = ex.map(feature_task, tasks, chunksize=8)
                X = np.array(list(results), np.float32)
        else:
            X = np.array([feature_task(t) for t in tasks], np.float32)
        return X
    ds = Tif3DDatasetSingle(
        root, records, args.initial, args.target, do_aug=aug, seed=args.seed
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        **({"multiprocessing_context": "spawn"} if args.workers else {}),
    )
    batches = []
    with torch.inference_mode():
        for i, (x, _, _) in enumerate(loader):
            features = model(x.to(args.device))
            if (
                not isinstance(features, torch.Tensor)
                or features.ndim != 2
                or features.shape[1] != 1024
            ):
                raise ValueError(
                    "Expected (batch,1024) frozen DINO embeddings; inspect external checkpoint/config/output"
                )
            batches.append(features.cpu().float().numpy())
            if i % 25 == 0:
                print(f"DINO {i*args.batch_size}/{len(ds)}", flush=True)
    return np.concatenate(batches)


def save(path, X, records, names):
    np.savez_compressed(
        path,
        embeddings=X,
        labels=np.array([r["label"] for r in records]),
        sample_ids=np.array([r["sample_id"] for r in records], dtype=str),
        row_ids=np.array([r.get("row_id", r["sample_id"]) for r in records], dtype=str),
        animals=np.array([r["animal"] for r in records], dtype=str),
        feature_names=np.array(names, dtype=str),
    )


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    table, meta = load_manifest(args.data)
    verify_patch_files(args.data, table)
    out = require_new(args.output)
    model = (
        load_dino(args.dino_repo, args.dino_config, args.dino_weights, args.device)
        if "dino" in args.families
        else None
    )
    records = table.to_dict("records")
    provenance = dict(
        class_names=meta.get("class_names"),
        manifest_sha256=sha256(args.data / "manifest.json"),
        initial=args.initial,
        target=args.target,
        levels=args.levels,
        seed=args.seed,
        families=args.families,
        environment=environment(),
        dino_weights_sha256=sha256(args.dino_weights) if model is not None else None,
        dino_config=str(args.dino_config),
        dino_repo=str(args.dino_repo),
    )
    if model is not None:
        proc = subprocess.run(
            ["git", "-C", str(args.dino_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        provenance["dino_git_revision"] = (
            proc.stdout.strip() if proc.returncode == 0 else "unavailable"
        )
        _, config_file, _ = validate_dino_paths(
            args.dino_repo, args.dino_config, args.dino_weights
        )
        provenance["dino_config_file"] = str(config_file)
        provenance["dino_config_sha256"] = sha256(config_file)
    write_json(out / "provenance.json", provenance)
    for family in args.families:
        dest = out / family
        dest.mkdir()
        names = (
            FEATURE_NAMES
            if family == "handcrafted"
            else [f"dino_{i}" for i in range(1024)]
        )
        print(f"Original {family}: {len(records)} nuclei", flush=True)
        X = extract(args.data, records, args, family, model)
        save(dest / "original.npz", X, records, names)
        for fold in selected_folds(meta, args.folds):
            foldout = dest / fold["name"]
            foldout.mkdir()
            plan = make_plan(table, fold, max(args.levels), args.seed)
            pd.DataFrame(plan)[
                ["row_id", "sample_id", "animal", "label", "draw", "augmentation_seed"]
            ].to_csv(foldout / "augmentation_plan.csv", index=False)
            print(f"{family} {fold['name']}: {len(plan)} augmented rows", flush=True)
            X = extract(args.data, plan, args, family, model, True)
            draws = np.array([r["draw"] for r in plan])
            for level in args.levels:
                idx = np.where(draws < level)[0]
                save(
                    foldout / f"aug_{level}.npz", X[idx], [plan[i] for i in idx], names
                )
    print("Feature extraction complete", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--families",
        nargs="+",
        choices=["dino", "handcrafted"],
        default=["dino", "handcrafted"],
    )
    p.add_argument("--levels", nargs="+", type=int, default=LEVELS)
    p.add_argument("--folds", nargs="+")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initial", type=int, nargs=3, default=[70] * 3)
    p.add_argument("--target", type=int, nargs=3, default=[112] * 3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for name in ["dino-repo", "dino-config", "dino-weights"]:
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    a.levels = sorted(set(a.levels))
    if min(a.levels) < 1 or min(a.initial + a.target) < 1:
        p.error("Positive sizes required")
    run(a)


if __name__ == "__main__":
    main()
