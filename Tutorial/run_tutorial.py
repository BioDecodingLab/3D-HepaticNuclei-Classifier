#!/usr/bin/env python3
"""Train or apply a frozen-3DINO hepatic nuclear morphotype classifier.

This tutorial is intentionally separate from the publication cross-validation
pipeline. It uses explicit training and independent test directories, while
reusing the repository's classifier grids, model fitting logic, metrics and
spatial preprocessing helpers where appropriate.

Run:
    python Tutorial/run_tutorial.py --config Tutorial/config.toml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import tifffile
import torch
from skimage.measure import regionprops
from skimage.transform import resize
from sklearn.model_selection import ParameterGrid, train_test_split
from torch.utils.data import DataLoader, Dataset

# Reuse the publication repository's tested downstream-model implementation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if not SCRIPTS_DIR.is_dir():
    raise RuntimeError(
        "Tutorial must remain inside the repository root so that ../scripts is available."
    )
sys.path.insert(0, str(SCRIPTS_DIR))

from classical_models import train_candidate  # noqa: E402
from dataset_helper import center_pad_3d, gaussian_blur_3d  # noqa: E402
from evaluation import export_evaluation, score_tuple  # noqa: E402
from model_grids import PARAM_GRIDS  # noqa: E402
from pipeline_common import LABELS, environment, sha256, stable_seed, write_json  # noqa: E402

LOGGER = logging.getLogger("nuclei_tutorial")
TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_id: str
    instance_id: int
    label: Optional[int]
    patch_path: str
    source_instance_path: str
    augmentation_seed: Optional[int] = None
    row_id: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Tutorial TOML config")
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    return cfg


def resolve_path(value: str | os.PathLike[str] | None, cfg: dict[str, Any]) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path(cfg["_config_dir"]) / p
    return p.resolve()


def validate_class_names(cfg: dict[str, Any]) -> dict[str, str]:
    names = {str(k): str(v).strip() for k, v in cfg.get("class_names", {}).items()}
    expected = {str(i) for i in LABELS}
    if set(names) != expected or any(not names[k] for k in expected):
        raise ValueError("[class_names] must define nonempty names for class IDs 1..5")
    if len(set(names.values())) != 5:
        raise ValueError("Class display names must be unique")
    return names


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ["run", "data", "training", "dino", "class_names"]:
        if section not in cfg:
            raise ValueError(f"Missing [{section}] section in config")

    run = cfg["run"]
    data = cfg["data"]
    tr = cfg["training"]
    dino = cfg["dino"]

    if tr["classifier"] not in PARAM_GRIDS:
        raise ValueError(f"Unknown classifier {tr['classifier']!r}; choose from {sorted(PARAM_GRIDS)}")
    if int(tr["augmentation_level"]) < 0:
        raise ValueError("augmentation_level must be >= 0")
    vf = float(tr["validation_fraction"])
    if not (0 < vf < 0.5):
        raise ValueError("validation_fraction must lie between 0 and 0.5")
    if dino["channel_strategy"] not in {"auto", "direct", "per_channel_concat"}:
        raise ValueError("dino.channel_strategy must be auto, direct, or per_channel_concat")
    if int(data["min_voxels"]) < 1 or int(data["border_margin"]) < 1:
        raise ValueError("min_voxels and border_margin must be positive")
    if len(data["initial_size"]) != 3 or len(data["target_size"]) != 3:
        raise ValueError("initial_size and target_size must each contain three integers")
    if min(map(int, data["initial_size"])) < 1 or min(map(int, data["target_size"])) < 1:
        raise ValueError("Spatial sizes must be positive")
    if float(data["normalization_pmax"]) <= float(data["normalization_pmin"]):
        raise ValueError("normalization_pmax must exceed normalization_pmin")
    if int(run["batch_size"]) < 1 or int(run["workers"]) < 0:
        raise ValueError("batch_size must be positive and workers must be >= 0")

    train_mode = bool(run["train_model"])
    train_dir = resolve_path(data.get("training_dir"), cfg)
    model_path = resolve_path(run.get("model_path"), cfg)
    if train_mode and train_dir is None:
        raise ValueError("data.training_dir is required when run.train_model = true")
    if not train_mode and model_path is None:
        raise ValueError("run.model_path is required when run.train_model = false")
    if not train_mode and resolve_path(data.get("test_dir"), cfg) is None and resolve_path(data.get("prediction_dir"), cfg) is None:
        raise ValueError("When train_model = false, provide data.test_dir and/or data.prediction_dir")
    validate_class_names(cfg)


def prepare_output(cfg: dict[str, Any]) -> Path:
    out = resolve_path(cfg["run"]["output_dir"], cfg)
    assert out is not None
    if out.exists() and any(out.iterdir()):
        if not bool(cfg["run"].get("overwrite_output", False)):
            raise FileExistsError(
                f"Output directory is nonempty: {out}. Set run.overwrite_output=true or choose a new output_dir."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def configure_logging(out: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log")]:
        h.setFormatter(formatter)
        LOGGER.addHandler(h)


def discover_tiffs(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in TIFF_SUFFIXES)
    result = {p.stem: p for p in files}
    if not result:
        raise FileNotFoundError(f"No TIFF files found in {folder}")
    if len(result) != len(files):
        raise ValueError(f"Duplicate TIFF filename stems in {folder}")
    return result


def infer_channel_axis(path: Path, arr: np.ndarray, setting: Any) -> int:
    if arr.ndim != 4:
        raise ValueError("Channel axis inference is only valid for 4D arrays")
    if isinstance(setting, int):
        axis = setting if setting >= 0 else arr.ndim + setting
        if not 0 <= axis < 4:
            raise ValueError(f"Invalid channel_axis={setting} for {path.name}")
        return axis
    if str(setting).lower() != "auto":
        raise ValueError("data.channel_axis must be an integer or 'auto'")

    try:
        with tifffile.TiffFile(path) as tf:
            axes = tf.series[0].axes
        candidates = [i for i, a in enumerate(axes) if a in {"C", "S"}]
        if len(candidates) == 1 and arr.shape[candidates[0]] in {1, 2, 3}:
            return candidates[0]
    except Exception:
        pass

    candidates = [i for i, n in enumerate(arr.shape) if n in {1, 2, 3}]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Cannot infer channel axis for {path.name} with shape {arr.shape}. "
        "Set data.channel_axis explicitly (e.g. 0 for CZYX or -1 for ZYXC)."
    )


def read_image_channels(path: Path, channel_axis_setting: Any) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[None, ...]
    elif arr.ndim == 4:
        axis = infer_channel_axis(path, arr, channel_axis_setting)
        arr = np.moveaxis(arr, axis, 0)
    else:
        raise ValueError(f"Expected 3D or 4D TIFF image, got {arr.shape} in {path}")
    if arr.shape[0] not in {1, 2, 3}:
        raise ValueError(f"Only 1, 2, or 3 image channels are supported; got {arr.shape[0]} in {path}")
    return arr.astype(np.float32, copy=False)


def normalize_channel(vol: np.ndarray, pmin: float, pmax: float) -> tuple[np.ndarray, float, float]:
    if not np.isfinite(vol).all():
        raise ValueError("Image contains nonfinite intensities")
    lo, hi = np.percentile(vol, [pmin, pmax])
    if hi <= lo:
        raise ValueError("Degenerate percentile normalization")
    out = np.clip((vol.astype(np.float32) - lo) / (hi - lo + 1e-30), 0.0, 1.0)
    return out.astype(np.float32), float(lo), float(hi)


def border_ids(labels: np.ndarray, margin: int) -> set[int]:
    border = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, margin)
        hi[axis] = slice(-margin, None)
        border[tuple(lo)] = True
        border[tuple(hi)] = True
    return set(map(int, np.unique(labels[border]))) - {0}


def preprocess_dataset(
    root: Path,
    cache_root: Path,
    cfg: dict[str, Any],
    labeled: bool,
    dataset_name: str,
) -> pd.DataFrame:
    data = cfg["data"]
    image_dir = root / data["image_subdir"]
    instance_dir = root / data["instance_subdir"]
    class_dir = root / data["class_subdir"]
    images = discover_tiffs(image_dir)
    instances = discover_tiffs(instance_dir)
    classes = discover_tiffs(class_dir) if labeled else {}

    if set(images) != set(instances):
        raise ValueError(f"Image and instance TIFF stems must match exactly in {root}")
    if labeled and set(images) != set(classes):
        raise ValueError(f"Image, instance and class TIFF stems must match exactly in {root}")

    cache_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    image_qc: list[dict[str, Any]] = []
    observed_raw_channels: set[int] = set()

    for image_id, image_path in images.items():
        LOGGER.info("Preprocessing %s image %s", dataset_name, image_id)
        image = read_image_channels(image_path, data["channel_axis"])
        observed_raw_channels.add(int(image.shape[0]))
        inst = tifffile.imread(instances[image_id])
        cls = tifffile.imread(classes[image_id]) if labeled else None

        if inst.ndim != 3 or not np.issubdtype(inst.dtype, np.integer) or inst.min() < 0:
            raise ValueError(f"Invalid 3D instance-label volume: {instances[image_id]}")
        if image.shape[1:] != inst.shape:
            raise ValueError(
                f"Spatial shape mismatch for {image_id}: image {image.shape[1:]}, instances {inst.shape}"
            )
        if labeled:
            assert cls is not None
            if cls.shape != inst.shape or not np.issubdtype(cls.dtype, np.integer):
                raise ValueError(f"Invalid class-label volume: {classes[image_id]}")
            if not set(np.unique(cls)) <= {0, *LABELS}:
                raise ValueError(f"Class map {classes[image_id]} must contain only integers 0..5")

        norm_channels = []
        norm_stats = []
        for c in range(image.shape[0]):
            ch, lo, hi = normalize_channel(
                image[c], float(data["normalization_pmin"]), float(data["normalization_pmax"])
            )
            norm_channels.append(ch)
            norm_stats.append((lo, hi))
        image = np.stack(norm_channels, axis=0)

        excluded = border_ids(inst, int(data["border_margin"]))
        n_saved = 0
        for region in regionprops(inst):
            sid = f"{image_id}__nucleus_{int(region.label)}"
            reason = ""
            if int(region.label) in excluded:
                reason = "border"
            elif int(region.area) < int(data["min_voxels"]):
                reason = "small"

            mask = region.image.astype(bool)
            label: Optional[int] = None
            majority_fraction = None
            labeled_fraction = None
            class_tie = None
            if labeled and not reason:
                assert cls is not None
                vals = cls[region.slice][mask]
                nonzero = vals[vals > 0]
                if not len(nonzero):
                    reason = "unclassified"
                else:
                    counts = np.bincount(nonzero.astype(int), minlength=6)
                    label = int(counts[1:].argmax() + 1)
                    majority_fraction = float(counts[label] / len(nonzero))
                    labeled_fraction = float(len(nonzero) / mask.sum())
                    class_tie = bool((counts[1:] == counts[label]).sum() > 1)

            if reason:
                qc_rows.append({"sample_id": sid, "image_id": image_id, "status": reason})
                continue

            patch = image[(slice(None),) + region.slice].copy()
            patch[:, ~mask] = 0.0
            patch_path = cache_root / image_id / f"{sid}.npz"
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(patch_path, image=patch.astype(np.float32), mask=mask.astype(np.uint8))
            rows.append(
                {
                    "sample_id": sid,
                    "image_id": image_id,
                    "instance_id": int(region.label),
                    "label": label,
                    "patch_path": str(patch_path),
                    "source_instance_path": str(instances[image_id].resolve()),
                    "raw_channels": int(image.shape[0]),
                    "mask_voxels": int(mask.sum()),
                }
            )
            qc_rows.append(
                {
                    "sample_id": sid,
                    "image_id": image_id,
                    "status": "saved",
                    "label": label,
                    "labeled_fraction": labeled_fraction,
                    "majority_fraction": majority_fraction,
                    "class_tie": class_tie,
                }
            )
            n_saved += 1

        image_qc.append(
            {
                "image_id": image_id,
                "raw_channels": int(image.shape[0]),
                "saved_nuclei": n_saved,
                "normalization_percentiles": f"{data['normalization_pmin']},{data['normalization_pmax']}",
                "channel_bounds": json.dumps(norm_stats),
            }
        )

    if len(observed_raw_channels) != 1:
        raise ValueError(
            f"All images within a dataset must use the same raw channel count; observed {sorted(observed_raw_channels)}"
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No valid nuclei remained after QC in {root}")
    table = table.sort_values("sample_id").reset_index(drop=True)
    table.to_csv(cache_root.parent / f"{dataset_name}_samples.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(cache_root.parent / f"{dataset_name}_qc.csv", index=False)
    pd.DataFrame(image_qc).to_csv(cache_root.parent / f"{dataset_name}_image_qc.csv", index=False)
    LOGGER.info("%s: retained %d nuclei from %d images", dataset_name, len(table), len(images))
    return table


def minmax_patch_channel(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32, copy=True)
    lo = float(out.min())
    hi = float(out.max())
    if hi <= lo:
        out.fill(0.0)
    else:
        out = (out - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    out[~mask] = 0.0
    return out


def standardize_patch(
    image: np.ndarray,
    mask: np.ndarray,
    initial: tuple[int, int, int],
    target: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 4 or mask.ndim != 3 or image.shape[1:] != mask.shape or not mask.any():
        raise ValueError("Expected aligned CxDxHxW image and nonempty 3D mask")
    mask_pad = center_pad_3d(mask.astype(bool), initial, only_pad=True)
    standardized = []
    for ch in image:
        ch = minmax_patch_channel(ch, mask)
        ch = center_pad_3d(ch, initial, only_pad=True)
        ch = resize(ch, target, order=0, preserve_range=True, anti_aliasing=True).astype(np.float32)
        standardized.append(ch)
    mask_out = resize(
        mask_pad.astype(np.uint8), target, order=0, preserve_range=True, anti_aliasing=False
    ).astype(bool)
    if not mask_out.any():
        raise ValueError("Mask disappeared during resize")
    x = np.stack(standardized, axis=0)
    x[:, ~mask_out] = 0.0
    return x.astype(np.float32), mask_out


def augment_multichannel(image: np.ndarray, mask: np.ndarray, seed: int, p: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    perm = perms[rng.randint(6)]
    image = image.transpose((0,) + tuple(a + 1 for a in perm)).copy()
    mask = mask.transpose(perm).copy()
    for axis in range(3):
        if rng.rand() < p:
            image = np.flip(image, axis=axis + 1).copy()
            mask = np.flip(mask, axis=axis).copy()

    if rng.rand() < p:
        scale = 1.0 + rng.uniform(-0.1, 0.1)
        shift = rng.uniform(-0.05, 0.05)
        image = image * scale + shift
    image = np.clip(image, 0.0, 1.0)
    image[:, ~mask] = 0.0

    blur_seed = int(rng.randint(0, 2**31 - 1))
    blurred = []
    for ch in image:
        local_rng = np.random.RandomState(blur_seed)
        blurred.append(
            gaussian_blur_3d(ch, local_rng, p=p, sigma_range=(0.4, 1.2), mask=mask)
        )
    image = np.stack(blurred, axis=0)

    if rng.rand() < p:
        image = image + rng.normal(0.0, 0.1, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    image[:, ~mask] = 0.0
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def build_dino_streams(raw_channels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return one stream for 1-channel input or three streams for 2/3-channel input.

    Two-channel data are converted to [C1, C2, normalize(C1+C2)]. Three-channel
    data are used as [C1, C2, C3]. This operation occurs after spatial
    standardization/augmentation so that the synthetic sum channel is derived
    from the actual transformed C1 and C2 data.
    """
    c = raw_channels.shape[0]
    if c == 1:
        streams = raw_channels
    elif c == 2:
        summed = raw_channels[0] + raw_channels[1]
        summed = minmax_patch_channel(summed, mask)
        streams = np.stack([raw_channels[0], raw_channels[1], summed], axis=0)
    elif c == 3:
        streams = raw_channels
    else:
        raise ValueError(f"Unsupported raw channel count: {c}")
    streams = np.clip(streams, 0.0, 1.0).astype(np.float32)
    streams[:, ~mask] = 0.0
    return streams


class PatchDataset(Dataset):
    def __init__(self, records: list[Sample], initial: tuple[int, int, int], target: tuple[int, int, int]):
        self.records = records
        self.initial = initial
        self.target = target

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        with np.load(r.patch_path, allow_pickle=False) as z:
            image = z["image"].astype(np.float32)
            mask = z["mask"].astype(bool)
        image, mask = standardize_patch(image, mask, self.initial, self.target)
        if r.augmentation_seed is not None:
            image, mask = augment_multichannel(image, mask, r.augmentation_seed)
        streams = build_dino_streams(image, mask)
        x = torch.from_numpy(streams * 2.0 - 1.0)
        return x, r.sample_id, (-1 if r.label is None else int(r.label)), r.image_id, int(r.instance_id)


def table_to_samples(table: pd.DataFrame) -> list[Sample]:
    result = []
    for row in table.to_dict("records"):
        label = row.get("label")
        if pd.isna(label):
            label = None
        result.append(
            Sample(
                sample_id=str(row["sample_id"]),
                image_id=str(row["image_id"]),
                instance_id=int(row["instance_id"]),
                label=None if label is None else int(label),
                patch_path=str(row["patch_path"]),
                source_instance_path=str(row["source_instance_path"]),
            )
        )
    return result


def load_dino(repo: Path, config: Path, weights: Path, device: str):
    repo = repo.resolve()
    weights = weights.resolve()
    if not (repo / "dinov2").is_dir():
        raise FileNotFoundError(f"No dinov2 source package in {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"3DINO weights not found: {weights}")
    if config.suffix != ".yaml":
        config = Path(str(config) + ".yaml")
    if not config.is_absolute() and not config.is_file():
        config = repo / "dinov2" / "configs" / config
    config = config.resolve()
    if not config.is_file():
        raise FileNotFoundError(f"3DINO configuration not found: {config}")
    if torch.device(device).type != "cuda":
        raise ValueError("The supported upstream 3DINO loader requires a CUDA device")

    cuda_device = torch.device(device)
    torch.cuda.set_device(cuda_device.index if cuda_device.index is not None else 0)
    sys.path.insert(0, str(repo))
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.eval.setup import build_model_for_eval

    model = build_model_for_eval(
        load_and_merge_config_3d(str(config.with_suffix(""))), str(weights)
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def infer_backbone_input_channels(model: torch.nn.Module) -> int:
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        proj = model.patch_embed.proj
        if hasattr(proj, "in_channels"):
            return int(proj.in_channels)
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            return int(m.in_channels)
    raise RuntimeError("Could not infer 3DINO patch-embedding input channels")


def resolve_channel_strategy(requested: str, model_in_channels: int, stream_count: int) -> str:
    if requested == "auto":
        if model_in_channels == stream_count:
            return "direct"
        if model_in_channels == 1:
            return "per_channel_concat"
        raise ValueError(
            f"3DINO expects {model_in_channels} channels but tutorial prepared {stream_count} streams. "
            "Use a compatible checkpoint/configuration."
        )
    if requested == "direct":
        if model_in_channels != stream_count:
            raise ValueError(
                f"Direct mode requested, but 3DINO expects {model_in_channels} channels and the input provides {stream_count}."
            )
        return requested
    if requested == "per_channel_concat":
        if model_in_channels != 1:
            raise ValueError("per_channel_concat requires a one-channel 3DINO backbone")
        return requested
    raise ValueError(requested)


def extract_embeddings(
    records: list[Sample],
    model: torch.nn.Module,
    strategy: str,
    initial: tuple[int, int, int],
    target: tuple[int, int, int],
    device: str,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    ds = PatchDataset(records, initial, target)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        **({"multiprocessing_context": "spawn"} if workers else {}),
    )
    chunks = []
    meta_rows = []
    with torch.inference_mode():
        for batch_i, (x, sample_ids, labels, image_ids, instance_ids) in enumerate(loader):
            if strategy == "direct":
                feat = model(x.to(device, non_blocking=True))
                if not isinstance(feat, torch.Tensor) or feat.ndim != 2:
                    raise ValueError("3DINO direct output must be a 2D tensor")
            elif strategy == "per_channel_concat":
                b, c, d, h, w = x.shape
                flat = x.reshape(b * c, 1, d, h, w).to(device, non_blocking=True)
                feat = model(flat)
                if not isinstance(feat, torch.Tensor) or feat.ndim != 2:
                    raise ValueError("3DINO per-channel output must be a 2D tensor")
                feat = feat.reshape(b, c * feat.shape[1])
            else:
                raise ValueError(strategy)
            if not torch.isfinite(feat).all():
                raise ValueError("Nonfinite 3DINO embedding detected")
            chunks.append(feat.detach().cpu().float().numpy())
            for sid, lab, img, inst in zip(sample_ids, labels.numpy(), image_ids, instance_ids.numpy()):
                meta_rows.append(
                    {
                        "sample_id": str(sid),
                        "label": None if int(lab) < 0 else int(lab),
                        "image_id": str(img),
                        "instance_id": int(inst),
                    }
                )
            if batch_i % 25 == 0:
                LOGGER.info("3DINO embedding progress: %d/%d nuclei", min((batch_i + 1) * batch_size, len(ds)), len(ds))
    X = np.concatenate(chunks, axis=0)
    if X.ndim != 2 or len(X) != len(records):
        raise RuntimeError("Embedding output shape mismatch")
    return X, pd.DataFrame(meta_rows)


def split_training_table(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = cfg["training"]
    vf = float(tr["validation_fraction"])
    if bool(tr.get("stratify_validation_by_image_and_class", True)):
        strata = table.image_id.astype(str) + ":" + table.label.astype(int).astype(str)
    else:
        strata = table.label.astype(int).astype(str)
    counts = strata.value_counts()
    nval = int(math.ceil(vf * len(table)))
    if counts.min() < 2 or nval < len(counts) or len(table) - nval < len(counts):
        raise ValueError(
            "The requested validation split cannot represent every stratum. Add more training nuclei/images, "
            "reduce stratification by setting training.stratify_validation_by_image_and_class=false, or adjust validation_fraction."
        )
    fit_ids, val_ids = train_test_split(
        table.sample_id.to_numpy(),
        test_size=vf,
        random_state=int(cfg["run"]["seed"]),
        stratify=strata,
    )
    fit = table[table.sample_id.isin(fit_ids)].copy()
    val = table[table.sample_id.isin(val_ids)].copy()
    for name, part in [("fit", fit), ("validation", val)]:
        if set(part.label.astype(int)) != set(LABELS):
            raise ValueError(f"{name} split does not contain all five classes")
    return fit.reset_index(drop=True), val.reset_index(drop=True)


def make_augmentation_plan(fit: pd.DataFrame, level: int, seed: int) -> list[Sample]:
    if level == 0:
        return table_to_samples(fit)
    rows: list[Sample] = []
    for image_id in sorted(fit.image_id.unique()):
        for label in LABELS:
            pool = fit[(fit.image_id == image_id) & (fit.label.astype(int) == int(label))]
            if pool.empty:
                raise ValueError(
                    f"Augmentation level {level} requires at least one fit nucleus for class {label} in training image {image_id}."
                )
            records = table_to_samples(pool.sort_values("sample_id"))
            rng = np.random.default_rng(stable_seed(seed, image_id, label, "tutorial_sampling"))
            for draw in range(level):
                source = records[int(rng.integers(len(records)))]
                aug_seed = stable_seed(seed, image_id, label, draw, "tutorial_augmentation")
                rows.append(
                    Sample(
                        sample_id=source.sample_id,
                        image_id=source.image_id,
                        instance_id=source.instance_id,
                        label=source.label,
                        patch_path=source.patch_path,
                        source_instance_path=source.source_instance_path,
                        augmentation_seed=aug_seed,
                        row_id=f"{image_id}__class{label}__draw{draw}",
                    )
                )
    return rows


def select_classifier(
    classifier_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    ids_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_original_fit: np.ndarray,
    y_original_fit: np.ndarray,
    ids_original_fit: np.ndarray,
    seed: int,
    search: bool,
    out: Path,
):
    grid = list(ParameterGrid(PARAM_GRIDS[classifier_name]))
    if not search:
        grid = [grid[0]]
        LOGGER.warning(
            "hyperparameter_search=false: using the first parameter combination from the publication grid."
        )
    rows = []
    best = None
    best_score = None
    best_params = None
    best_metrics = None
    for i, params in enumerate(grid, start=1):
        LOGGER.info("Fitting %s candidate %d/%d", classifier_name, i, len(grid))
        model, metrics, history = train_candidate(
            classifier_name,
            params,
            X_train,
            y_train,
            X_val,
            y_val,
            ids_train,
            X_original_fit,
            y_original_fit,
            ids_original_fit,
            seed=seed,
        )
        score = score_tuple(metrics)
        rows.append({"candidate": i, **params, **metrics, "score_tuple": repr(score)})
        if best_score is None or score > best_score:
            best_score = score
            best = model
            best_params = params
            best_metrics = metrics
        if history:
            pd.DataFrame(history).to_csv(out / f"candidate_{i:03d}_mlp_history.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "validation_search.csv", index=False)
    if best is None:
        raise RuntimeError("No classifier candidate was fitted")
    return best, best_params, best_metrics


def save_embedding_npz(path: Path, X: np.ndarray, meta: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        embeddings=X.astype(np.float32),
        sample_ids=meta.sample_id.astype(str).to_numpy(),
        labels=np.array([-1 if pd.isna(v) else int(v) for v in meta.label], dtype=np.int16),
        image_ids=meta.image_id.astype(str).to_numpy(),
        instance_ids=meta.instance_id.astype(np.int64).to_numpy(),
    )


def model_artifact_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / "model.joblib", path / "model_metadata.json"
    if path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
        return path, path.with_name("model_metadata.json")
    raise ValueError("model_path must be a trained-model directory or a .joblib/.pkl file")


def predict_table(model, X: np.ndarray, meta: pd.DataFrame, class_names: dict[str, str]) -> pd.DataFrame:
    pred = model.predict(X).astype(int)
    prob = model.predict_proba(X)
    if prob.shape != (len(X), 5):
        raise ValueError(f"Expected five class probabilities, got {prob.shape}")
    frame = meta.copy()
    frame["predicted_label"] = pred
    frame["predicted_class"] = [class_names[str(int(v))] for v in pred]
    for i, c in enumerate(LABELS):
        frame[f"p_{c}"] = prob[:, i]
    return frame


def write_predicted_class_maps(predictions: pd.DataFrame, sample_table: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lookup = sample_table.set_index("sample_id")
    merged = predictions.join(
        lookup[["source_instance_path"]], on="sample_id", how="left", validate="one_to_one"
    )
    for image_id, group in merged.groupby("image_id"):
        paths = group.source_instance_path.unique()
        if len(paths) != 1:
            raise ValueError(f"Inconsistent instance source for image {image_id}")
        instances = tifffile.imread(paths[0])
        max_id = int(instances.max())
        lut = np.zeros(max_id + 1, dtype=np.uint8)
        for row in group.itertuples(index=False):
            if int(row.instance_id) <= max_id:
                lut[int(row.instance_id)] = np.uint8(int(row.predicted_label))
        class_map = lut[instances.astype(np.int64)]
        tifffile.imwrite(out / f"{image_id}_predicted_classes.tif", class_map, metadata={"axes": "ZYX"})


def git_revision(repo: Path) -> str:
    try:
        p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def run() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    out = prepare_output(cfg)
    configure_logging(out)

    seed = int(cfg["run"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    class_names = validate_class_names(cfg)
    data_cfg = cfg["data"]
    initial = tuple(map(int, data_cfg["initial_size"]))
    target = tuple(map(int, data_cfg["target_size"]))
    device = str(cfg["run"]["device"])
    batch_size = int(cfg["run"]["batch_size"])
    workers = int(cfg["run"]["workers"])

    dino_repo = resolve_path(cfg["dino"]["repo"], cfg)
    dino_config = resolve_path(cfg["dino"]["config"], cfg)
    dino_weights = resolve_path(cfg["dino"]["weights"], cfg)
    assert dino_repo is not None and dino_config is not None and dino_weights is not None
    LOGGER.info("Loading frozen 3DINO backbone")
    dino, resolved_dino_config = load_dino(dino_repo, dino_config, dino_weights, device)
    backbone_in_channels = infer_backbone_input_channels(dino)
    LOGGER.info("3DINO patch embedding expects %d input channel(s)", backbone_in_channels)

    cache = out / "cache"
    train_mode = bool(cfg["run"]["train_model"])
    model = None
    model_metadata: dict[str, Any] = {}
    resolved_strategy = None
    expected_raw_channels = None

    if train_mode:
        train_dir = resolve_path(data_cfg["training_dir"], cfg)
        assert train_dir is not None
        train_table = preprocess_dataset(train_dir, cache / "train_patches", cfg, True, "train")
        expected_raw_channels = int(train_table.raw_channels.iloc[0])
        stream_count = 1 if expected_raw_channels == 1 else 3
        resolved_strategy = resolve_channel_strategy(
            str(cfg["dino"]["channel_strategy"]), backbone_in_channels, stream_count
        )
        LOGGER.info(
            "Training raw channels=%d -> 3DINO streams=%d; resolved strategy=%s",
            expected_raw_channels,
            stream_count,
            resolved_strategy,
        )

        fit_table, val_table = split_training_table(train_table, cfg)
        pd.DataFrame(
            [
                *({"sample_id": s, "split": "fit"} for s in fit_table.sample_id),
                *({"sample_id": s, "split": "validation"} for s in val_table.sample_id),
            ]
        ).to_csv(out / "train_validation_split.csv", index=False)

        LOGGER.info("Extracting original fit embeddings")
        X_fit_orig, m_fit_orig = extract_embeddings(
            table_to_samples(fit_table), dino, resolved_strategy, initial, target, device, batch_size, workers
        )
        LOGGER.info("Extracting validation embeddings")
        X_val, m_val = extract_embeddings(
            table_to_samples(val_table), dino, resolved_strategy, initial, target, device, batch_size, workers
        )

        level = int(cfg["training"]["augmentation_level"])
        train_records = make_augmentation_plan(fit_table, level, seed)
        if level == 0:
            X_train = X_fit_orig
            m_train = m_fit_orig
        else:
            LOGGER.info("Extracting %d augmented training embeddings", len(train_records))
            X_train, m_train = extract_embeddings(
                train_records, dino, resolved_strategy, initial, target, device, batch_size, workers
            )
            pd.DataFrame(
                [
                    {
                        "row_id": r.row_id,
                        "sample_id": r.sample_id,
                        "image_id": r.image_id,
                        "label": r.label,
                        "augmentation_seed": r.augmentation_seed,
                    }
                    for r in train_records
                ]
            ).to_csv(out / "augmentation_plan.csv", index=False)

        classifier_out = out / "classifier_search"
        classifier_out.mkdir(parents=True, exist_ok=True)
        model, best_params, best_metrics = select_classifier(
            str(cfg["training"]["classifier"]),
            X_train,
            m_train.label.astype(int).to_numpy(),
            m_train.sample_id.astype(str).to_numpy(),
            X_val,
            m_val.label.astype(int).to_numpy(),
            X_fit_orig,
            m_fit_orig.label.astype(int).to_numpy(),
            m_fit_orig.sample_id.astype(str).to_numpy(),
            seed,
            bool(cfg["training"].get("hyperparameter_search", True)),
            classifier_out,
        )

        model_dir = out / "trained_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "model.joblib"
        joblib.dump(model, model_file)
        model_metadata = {
            "schema": "nuclei_tutorial_model_v1",
            "classifier": str(cfg["training"]["classifier"]),
            "classifier_params": best_params,
            "validation_metrics": best_metrics,
            "augmentation_level": level,
            "class_names": class_names,
            "raw_channel_count": expected_raw_channels,
            "dino_stream_count": stream_count,
            "channel_strategy_requested": str(cfg["dino"]["channel_strategy"]),
            "channel_strategy_resolved": resolved_strategy,
            "embedding_dim": int(X_train.shape[1]),
            "initial_size": list(initial),
            "target_size": list(target),
            "normalization_percentiles": [float(data_cfg["normalization_pmin"]), float(data_cfg["normalization_pmax"])],
            "min_voxels": int(data_cfg["min_voxels"]),
            "border_margin": int(data_cfg["border_margin"]),
            "seed": seed,
            "dino_backbone_input_channels": backbone_in_channels,
            "dino_weights_sha256": sha256(dino_weights),
            "dino_config": str(resolved_dino_config),
            "dino_config_sha256": sha256(resolved_dino_config),
            "dino_git_revision": git_revision(dino_repo),
            "model_sha256": sha256(model_file),
        }
        write_json(model_dir / "model_metadata.json", model_metadata)
        LOGGER.info("Saved trained model to %s", model_file)

        if bool(cfg["run"].get("save_embeddings", True)):
            save_embedding_npz(out / "fit_original_embeddings.npz", X_fit_orig, m_fit_orig)
            save_embedding_npz(out / "validation_embeddings.npz", X_val, m_val)
    else:
        model_path = resolve_path(cfg["run"]["model_path"], cfg)
        assert model_path is not None
        model_file, metadata_file = model_artifact_paths(model_path)
        if not model_file.is_file() or not metadata_file.is_file():
            raise FileNotFoundError(
                f"Expected model and metadata files: {model_file} and {metadata_file}"
            )
        model = joblib.load(model_file)
        model_metadata = json.loads(metadata_file.read_text())
        if model_metadata.get("schema") != "nuclei_tutorial_model_v1":
            raise ValueError("Unsupported tutorial model metadata schema")
        if model_metadata.get("dino_weights_sha256") != sha256(dino_weights):
            raise ValueError("Configured 3DINO weights differ from the weights recorded with the trained model")
        if model_metadata.get("dino_config_sha256") != sha256(resolved_dino_config):
            raise ValueError("Configured 3DINO config differs from the config recorded with the trained model")
        class_names = {str(k): str(v) for k, v in model_metadata["class_names"].items()}
        expected_raw_channels = int(model_metadata["raw_channel_count"])
        resolved_strategy = str(model_metadata["channel_strategy_resolved"])
        if int(model_metadata["dino_backbone_input_channels"]) != backbone_in_channels:
            raise ValueError("Loaded 3DINO backbone input-channel count differs from the trained model metadata")
        LOGGER.info("Loaded trained classifier from %s", model_file)

    assert model is not None and resolved_strategy is not None and expected_raw_channels is not None

    # Independent labeled test evaluation.
    test_dir = resolve_path(data_cfg.get("test_dir"), cfg)
    if test_dir is not None:
        test_table = preprocess_dataset(test_dir, cache / "test_patches", cfg, True, "test")
        raw_channels = int(test_table.raw_channels.iloc[0])
        if raw_channels != expected_raw_channels:
            raise ValueError(
                f"Test data use {raw_channels} raw channels, but the trained model expects {expected_raw_channels}."
            )
        LOGGER.info("Extracting independent test embeddings")
        X_test, m_test = extract_embeddings(
            table_to_samples(test_table), dino, resolved_strategy, initial, target, device, batch_size, workers
        )
        if X_test.shape[1] != int(model_metadata["embedding_dim"]):
            raise ValueError("Test embedding dimension does not match trained model")
        y_test = m_test.label.astype(int).to_numpy()
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)
        test_out = out / "test_evaluation"
        metrics = export_evaluation(
            test_out,
            y_test,
            pred,
            prob,
            m_test.sample_id.astype(str).to_numpy(),
            m_test.image_id.astype(str).to_numpy(),
            title="Independent test",
            class_names=class_names,
        )
        test_predictions = predict_table(model, X_test, m_test, class_names)
        test_predictions.to_csv(test_out / "predictions_with_instance_ids.csv", index=False)
        write_predicted_class_maps(test_predictions, test_table, test_out / "predicted_class_volumes")
        if bool(cfg["run"].get("save_embeddings", True)):
            save_embedding_npz(test_out / "embeddings.npz", X_test, m_test)
        LOGGER.info("Independent test macro-F1: %.5f", metrics["f1_macro"])

    # Unlabeled prediction on new segmented images.
    prediction_dir = resolve_path(data_cfg.get("prediction_dir"), cfg)
    if prediction_dir is not None:
        pred_table = preprocess_dataset(
            prediction_dir, cache / "prediction_patches", cfg, False, "prediction"
        )
        raw_channels = int(pred_table.raw_channels.iloc[0])
        if raw_channels != expected_raw_channels:
            raise ValueError(
                f"Prediction data use {raw_channels} raw channels, but the trained model expects {expected_raw_channels}."
            )
        LOGGER.info("Extracting prediction embeddings")
        X_pred, m_pred = extract_embeddings(
            table_to_samples(pred_table), dino, resolved_strategy, initial, target, device, batch_size, workers
        )
        if X_pred.shape[1] != int(model_metadata["embedding_dim"]):
            raise ValueError("Prediction embedding dimension does not match trained model")
        prediction_out = out / "predictions"
        prediction_out.mkdir(parents=True, exist_ok=True)
        predictions = predict_table(model, X_pred, m_pred, class_names)
        predictions.to_csv(prediction_out / "predictions.csv", index=False)
        write_predicted_class_maps(predictions, pred_table, prediction_out / "predicted_class_volumes")
        if bool(cfg["run"].get("save_embeddings", True)):
            save_embedding_npz(prediction_out / "embeddings.npz", X_pred, m_pred)
        LOGGER.info("Saved predictions for %d nuclei", len(predictions))

    provenance = {
        "config_file": str(Path(cfg["_config_path"]).resolve()),
        "environment": environment(),
        "class_names": class_names,
        "train_model": train_mode,
        "resolved_channel_strategy": resolved_strategy,
        "expected_raw_channels": expected_raw_channels,
        "dino_repo": str(dino_repo),
        "dino_config": str(resolved_dino_config),
        "dino_weights": str(dino_weights),
        "dino_weights_sha256": sha256(dino_weights),
        "dino_git_revision": git_revision(dino_repo),
    }
    write_json(out / "run_metadata.json", provenance)
    write_json(out / "resolved_model_metadata.json", model_metadata)
    LOGGER.info("Tutorial workflow completed successfully: %s", out)


if __name__ == "__main__":
    run()
