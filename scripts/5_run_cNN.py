#!/usr/bin/env python3
"""Direct 3D models with shared 10% validation and validation-selected checkpoints."""

import argparse
import importlib
import itertools
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from pipeline_common import (
    load_manifest,
    class_mapping,
    selected_folds,
    require_new,
    write_json,
    read_json,
    stable_seed,
    sha256,
    environment,
    verify_patch_files,
)
from dataset_helper import Tif3DDatasetSingle
from direct_models import SmallResNet3D, DinoClassifier
from evaluation import metrics, score_tuple, export_evaluation


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build(name, p, a):
    if name == "resnet3d_18":
        return SmallResNet3D(num_classes=5, base_channels=32, dropout=p["dropout"])
    backbone = importlib.import_module("2_embedding_extraction").load_dino(
        a.dino_repo, a.dino_config, a.dino_weights, a.device
    )
    model = DinoClassifier(backbone, 5, p["hidden_dim"], p["dropout"])
    for param in backbone.parameters():
        param.requires_grad_(not p["freeze_backbone"])
    return model


def loader(ds, batch, workers, shuffle, seed):
    # Nonpersistent workers see the new epoch set on the parent dataset.
    return DataLoader(
        ds,
        batch_size=batch,
        num_workers=workers,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=False,
    )


def predict(model, ds, a):
    model.eval()
    ys = []
    probs = []
    ids = []
    with torch.inference_mode():
        for x, y, sid in loader(ds, a.batch_size, a.workers, False, a.seed):
            with torch.autocast(
                device_type=torch.device(a.device).type,
                enabled=str(a.device).startswith("cuda"),
            ):
                logits = model(x.to(a.device))
            probs.append(logits.float().softmax(1).cpu().numpy())
            ys.extend((y.numpy() + 1).tolist())
            ids.extend(sid)
    prob = np.concatenate(probs)
    return np.array(ys), prob.argmax(1) + 1, prob, np.array(ids)


def train_once(model, params, train, val, a, seed, dest):
    y = np.array([r["label"] - 1 for r in train.records])
    counts = np.bincount(y, minlength=5)
    if (counts == 0).any():
        raise ValueError("Missing training class")
    weights = torch.tensor(len(y) / (5 * counts), dtype=torch.float32, device=a.device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
    model.to(a.device)
    if (
        str(a.device).startswith("cuda")
        and torch.cuda.device_count() > 1
        and a.data_parallel
    ):
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=str(a.device).startswith("cuda"))
    bestscore = None
    best = None
    stale = 0
    history = []
    for epoch in range(1, a.epochs + 1):
        train.set_epoch(epoch)
        model.train()
        plain = model.module if isinstance(model, nn.DataParallel) else model
        if params.get("freeze_backbone"):
            plain.feature_extractor.backbone.eval()
        total = 0.0
        denominator = 0.0
        for x, y, _ in loader(
            train, a.batch_size, a.workers, True, stable_seed(seed, epoch)
        ):
            x, y = x.to(a.device), y.to(a.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch.device(a.device).type,
                enabled=str(a.device).startswith("cuda"),
            ):
                logits = model(x)
                loss_sum = criterion(logits, y)
                denom = weights[y].sum()
                loss = loss_sum / denom
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CNN loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss_sum.detach())
            denominator += float(denom)
        yt, yp, prob, _ = predict(model, val, a)
        m = metrics(yt, yp, prob)
        history.append(
            dict(
                epoch=epoch,
                train_weighted_loss=total / denominator,
                lr=optimizer.param_groups[0]["lr"],
                **m,
            )
        )
        print(f'Epoch {epoch}: validation macro-F1={m["f1_macro"]:.5f}', flush=True)
        scheduler.step(m["f1_macro"])
        if bestscore is None or score_tuple(m) > bestscore:
            bestscore = score_tuple(m)
            stale = 0
            best = dict(epoch=epoch, metrics=m)
            state = {k: v.detach().cpu().clone() for k, v in plain.state_dict().items()}
            torch.save(
                dict(
                    state_dict=state,
                    params=params,
                    epoch=epoch,
                    model=a.model,
                    initial=a.initial,
                    target=a.target,
                    seed=seed,
                    classes=[1, 2, 3, 4, 5],
                ),
                dest / "checkpoint.pt",
            )
        else:
            stale += 1
        if stale >= a.patience:
            break
    pd.DataFrame(history).to_csv(dest / "training_history.csv", index=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def datasets(table, fold, a):
    lookup = table.set_index("sample_id", drop=False)
    train = Tif3DDatasetSingle(
        a.data,
        lookup.loc[fold["train"]].to_dict("records"),
        a.initial,
        a.target,
        do_aug=True,
        seed=a.seed,
        epoch_augmentation=True,
    )
    val = Tif3DDatasetSingle(
        a.data, lookup.loc[fold["validation"]].to_dict("records"), a.initial, a.target
    )
    return train, val


def select(a):
    table, meta = load_manifest(a.data)
    verify_patch_files(a.data, table)
    out = require_new(a.output)
    grid = {"dropout": [0.2, 0.4], "lr": [1e-4, 3e-4], "weight_decay": [1e-4, 1e-3]}
    if a.model == "dino_cls":
        grid.update(
            hidden_dim=[256, 512], freeze_backbone=[True, False], lr=[1e-4, 5e-4]
        )
    grid = list(ParameterGrid(grid))
    grid = grid[:1] if a.smoke else grid
    selected = []
    for fold in selected_folds(meta, a.folds):
        train, val = datasets(table, fold, a)
        records = []
        best = None
        for i, p in enumerate(grid):
            # Same per-fold initialization seed across candidates, independent of execution order.
            seed = stable_seed(a.seed, fold["name"], "direct")
            seed_all(seed)
            dest = out / fold["name"] / f"candidate_{i:03d}"
            dest.mkdir(parents=True)
            model = build(a.model, p, a)
            result = train_once(model, p, train, val, a, seed, dest)
            record = dict(
                fold=fold["name"],
                candidate=i,
                params=p,
                **result,
                artifact=str((dest / "checkpoint.pt").relative_to(out)),
                sha256=sha256(dest / "checkpoint.pt"),
            )
            records.append(record)
            if best is None or score_tuple(record["metrics"]) > score_tuple(
                best["metrics"]
            ):
                best = record
        write_json(out / fold["name"] / "search_results.json", records)
        selected.append(best)
    write_json(
        out / "selection.json",
        dict(
            selection=selected,
            class_names=class_mapping(meta),
            model=a.model,
            seed=a.seed,
            initial=a.initial,
            target=a.target,
            epochs=a.epochs,
            patience=a.patience,
            batch_size=a.batch_size,
            smoke_test=a.smoke,
            manifest_sha256=sha256(a.data / "manifest.json"),
            dino_weights_sha256=(
                sha256(a.dino_weights) if a.model == "dino_cls" else None
            ),
            environment=environment(),
        ),
    )


def evaluate(a):
    table, meta = load_manifest(a.data)
    verify_patch_files(a.data, table)
    lock = read_json(a.output / "selection.json")
    if lock["manifest_sha256"] != sha256(a.data / "manifest.json"):
        raise ValueError("Manifest changed")
    a.model = lock["model"]
    a.initial = lock["initial"]
    a.target = lock["target"]
    a.batch_size = lock["batch_size"]
    if a.model == "dino_cls" and sha256(a.dino_weights) != lock["dino_weights_sha256"]:
        raise ValueError("DINO weights changed")
    lookup = table.set_index("sample_id", drop=False)
    foldmap = {f["name"]: f for f in meta["folds"]}
    rows = []
    for r in lock["selection"]:
        path = a.output / r["artifact"]
        if sha256(path) != r["sha256"]:
            raise ValueError("Checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = build(a.model, r["params"], a)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(a.device)
        fold = foldmap[r["fold"]]
        for split in ["validation", "test"]:
            records = lookup.loc[fold[split]].to_dict("records")
            ds = Tif3DDatasetSingle(a.data, records, a.initial, a.target)
            y, pred, prob, ids = predict(model, ds, a)
            m = export_evaluation(
                a.output / r["fold"] / split,
                y,
                pred,
                prob,
                ids,
                lookup.loc[ids, "animal"].to_numpy(),
                f"{a.model} {r['fold']} {split}",
                class_names=class_mapping(lock),
            )
            if split == "test":
                rows.append(
                    dict(
                        family=a.model,
                        fold=r["fold"],
                        model=a.model,
                        level="direct",
                        selected=True,
                        **m,
                    )
                )
        del model
    pd.DataFrame(rows).to_csv(a.output / "test_metrics_all_conditions.csv", index=False)
    write_json(
        a.output / "evaluation_provenance.json",
        dict(
            selection_sha256=sha256(a.output / "selection.json"),
            smoke_test=lock["smoke_test"],
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["select", "evaluate"], required=True)
    for n in ["data", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument(
        "--model", choices=["resnet3d_18", "dino_cls"], default="resnet3d_18"
    )
    p.add_argument("--folds", nargs="+")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-parallel", action="store_true")
    p.add_argument("--initial", type=int, nargs=3, default=[70] * 3)
    p.add_argument("--target", type=int, nargs=3, default=[112] * 3)
    for n in ["dino-repo", "dino-config", "dino-weights"]:
        p.add_argument("--" + n, type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.epochs = 1
    if min(a.epochs, a.patience, a.batch_size) < 1:
        p.error("Positive training settings required")
    seed_all(a.seed)
    (select if a.phase == "select" else evaluate)(a)


if __name__ == "__main__":
    main()
