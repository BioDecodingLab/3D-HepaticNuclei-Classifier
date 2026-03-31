#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct 3D patch classification from TIFF volumes using:
1) 3D DINO backbone + trainable classification head
2) 3D ResNet classifier

Cross-validation is folder-based:
patches_dir/
    img_1/
        label_1/*.tif
        label_2/*.tif
        ...
    img_2/
        ...
Each img_* folder is one held-out test fold.

This script assumes you already have available:
- normalize_image
- center_pad_3d
- augment_3dimage
- load_and_merge_config_3d
- build_model_for_eval

and that your dataset class generation logic should remain unchanged.
"""
import sys
sys.path.append("/medicina/hmorales/projects/Nuclei3DClassification/code/3DINO")

import os
import gc
import glob
import json
import math
import copy
import time
import random
from pathlib import Path
from collections import Counter



import numpy as np
import pandas as pd
import tifffile as tiff
import skimage.transform
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.cuda.amp import autocast
from torch.amp import GradScaler
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit

from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config_3d
from dataset_helper import *

# =============================================================================
# USER CONFIG
# =============================================================================

SEED = 42

patches_dir  = Path("/medicina/hmorales/projects/Nuclei3DClassification/data/patches/")
labels = [1, 2, 3, 4, 5]
target_DHW0 = (56,56,56)
target_DHW  = (112, 112, 112)

batch_size = 256
num_workers = 48

config_file = "/medicina/hmorales/projects/Nuclei3DClassification/code/3DINO/dinov2/configs/train/vit3d_highres"
pretrained_weights = "/medicina/hmorales/projects/Nuclei3DClassification/data/3dino_vit_weights.pth"

output_root = Path("/medicina/hmorales/projects/Nuclei3DClassification/results/direct_patch_classification")

CLASS_NAMES = [f"label_{x}" for x in labels]
LABEL_ORDER = labels
NUM_CLASSES = len(labels)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def collect_fold_dirs(patches_root: Path):
    fold_dirs = sorted([p for p in patches_root.iterdir() if p.is_dir() and p.name.startswith("img_")])
    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No img_* folders found in: {patches_root}")
    return fold_dirs


def build_dataset_for_folders(folder_list, do_aug=False, target_per_label=None, seed=SEED):
    datasets = []
    for fold_dir in folder_list:
        ds = Tif3DDatasetSingle(
            base_dir=str(fold_dir),
            labels=labels,
            target_dhw0=target_DHW0,
            target_dhw=target_DHW,
            target_per_label=target_per_label,
            do_aug=do_aug,
            seed=seed,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def extract_labels_from_dataset(dataset):
    ys = []
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            ys.extend([y for _, y in ds.items])
    else:
        ys.extend([y for _, y in dataset.items])
    return np.asarray(ys, dtype=int)


def make_train_val_subsets(dataset, val_fraction=0.10, random_state=SEED):
    y_all = extract_labels_from_dataset(dataset)

    idx_all = np.arange(len(y_all))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(splitter.split(idx_all, y_all))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    return train_subset, val_subset


def subset_labels(subset):
    if not isinstance(subset, Subset):
        raise TypeError("Expected a torch.utils.data.Subset")
    base = subset.dataset
    labels_out = []

    if isinstance(base, ConcatDataset):
        cumulative = np.cumsum([0] + [len(ds) for ds in base.datasets])
        for idx in subset.indices:
            ds_id = np.searchsorted(cumulative, idx, side="right") - 1
            local_idx = idx - cumulative[ds_id]
            _, y = base.datasets[ds_id].items[local_idx]
            labels_out.append(y)
    else:
        for idx in subset.indices:
            _, y = base.items[idx]
            labels_out.append(y)

    return np.asarray(labels_out, dtype=int)


def compute_class_weights(y, label_order):
    counts = Counter(y.tolist())
    n = len(y)
    k = len(label_order)
    weights = []
    for c in label_order:
        count_c = counts.get(c, 1)
        weights.append(n / (k * count_c))
    return torch.tensor(weights, dtype=torch.float32)


def remap_labels_to_zero_based(y_np, label_order):
    mapping = {lab: i for i, lab in enumerate(label_order)}
    return np.asarray([mapping[v] for v in y_np], dtype=np.int64)


def remap_tensor_labels(y_tensor, label_order):
    mapping = {lab: i for i, lab in enumerate(label_order)}
    out = torch.empty_like(y_tensor, dtype=torch.long)
    for src, dst in mapping.items():
        out[y_tensor == src] = dst
    return out


def save_confusion_matrix(y_true, y_pred, out_path, title, normalize=None, class_names=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)) if class_names else None, normalize=normalize)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize is not None else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, format(val, fmt),
                     ha="center", va="center",
                     color="white" if val > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def compute_metrics(y_true_zero, y_pred_zero, prefix_name="Test"):
    acc = accuracy_score(y_true_zero, y_pred_zero)
    bal_acc = balanced_accuracy_score(y_true_zero, y_pred_zero)

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true_zero, y_pred_zero, average="weighted", zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true_zero, y_pred_zero, average="macro", zero_division=0
    )

    per_class_prec, per_class_rec, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_true_zero, y_pred_zero, labels=np.arange(NUM_CLASSES), average=None, zero_division=0
    )

    per_class_df = pd.DataFrame({
        "class_id_zero_based": np.arange(NUM_CLASSES),
        "class_name": CLASS_NAMES,
        "precision": per_class_prec,
        "recall": per_class_rec,
        "f1_score": per_class_f1,
        "support": per_class_sup,
    })

    report = classification_report(
        y_true_zero, y_pred_zero,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "balanced_accuracy": float(bal_acc),
    }

    return metrics, per_class_df, report


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_planes, planes,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SmallResNet3D(nn.Module):
    def __init__(self, num_classes=5, in_channels=1, base_channels=32, dropout=0.2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, num_classes)

        self._init_weights()

    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = [BasicBlock3D(in_planes, planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_resnet3d_classifier(num_classes, dropout=0.2, base_channels=32):
    return SmallResNet3D(
        num_classes=num_classes,
        in_channels=1,
        base_channels=base_channels,
        dropout=dropout
    )

class Identity(nn.Module):
    def forward(self, x):
        return x


class DINOBackboneWrapper(nn.Module):
    """
    Wraps your DINO model and extracts a feature tensor.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)

        # common possibilities
        if torch.is_tensor(out):
            feats = out
        elif isinstance(out, dict):
            for key in ["x_norm_clstoken", "x_cls", "cls_token", "features", "embeddings", "x"]:
                if key in out and torch.is_tensor(out[key]):
                    feats = out[key]
                    break
            else:
                raise TypeError(f"Could not find tensor features in DINO dict output keys: {list(out.keys())}")
        elif isinstance(out, (list, tuple)):
            tensor_candidates = [z for z in out if torch.is_tensor(z)]
            if len(tensor_candidates) == 0:
                raise TypeError("DINO returned list/tuple without tensor outputs.")
            feats = tensor_candidates[0]
        else:
            raise TypeError(f"Unsupported DINO output type: {type(out)}")

        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)

        return feats

class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.feature_extractor = DINOBackboneWrapper(backbone)

        if hasattr(backbone, "embed_dim"):
            feat_dim = backbone.embed_dim
        elif hasattr(backbone, "num_features"):
            feat_dim = backbone.num_features
        else:
            raise AttributeError(
                "Could not infer feature dimension from backbone. "
                "Expected attribute 'embed_dim' or 'num_features'."
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        return logits

class DinoClassifierOld(nn.Module):
    def __init__(self, backbone, num_classes, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.feature_extractor = DINOBackboneWrapper(backbone)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *target_DHW)
            dummy_feats = self.feature_extractor(dummy)
            feat_dim = dummy_feats.shape[1]

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        return logits


def build_resnet3d_classifier(num_classes):
    model = r3d_18(weights=None)

    # adapt first conv from 3 channels to 1 channel
    old_conv = model.stem[0]
    model.stem[0] = nn.Conv3d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_dino_backbone(config_file, pretrained_weights):
    # EXACTLY as requested
    cfg = load_and_merge_config_3d(config_file)
    model = build_model_for_eval(cfg, pretrained_weights)
    model = model.to(DEVICE)
    return model


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True


def build_model(model_name, params):
    if model_name == "dino_cls":
        backbone = load_dino_backbone(config_file, pretrained_weights)
        model = DinoClassifier(
            backbone=backbone,
            num_classes=NUM_CLASSES,
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
        )

        if params.get("freeze_backbone", False):
            freeze_module(model.feature_extractor.backbone)
        else:
            unfreeze_module(model.feature_extractor.backbone)

        return model

    elif model_name == "resnet3d_18":
        return build_resnet3d_classifier(
            num_classes=NUM_CLASSES,
            dropout=params.get("dropout", 0.2),
            base_channels=params.get("base_channels", 32),
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# =============================================================================
# TRAIN / EVAL
# =============================================================================

@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()

    all_logits = []
    all_probs = []
    all_preds = []
    all_true = []
    all_paths = []

    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y0 = remap_tensor_labels(y, LABEL_ORDER)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(x)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_logits.append(logits.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_true.append(y0.detach().cpu())
        all_paths.extend(list(paths))

    return {
        "logits": torch.cat(all_logits, dim=0).numpy(),
        "probs": torch.cat(all_probs, dim=0).numpy(),
        "preds": torch.cat(all_preds, dim=0).numpy(),
        "true": torch.cat(all_true, dim=0).numpy(),
        "paths": np.asarray(all_paths, dtype=str),
    }


def run_one_epoch_train(model, loader, optimizer, criterion, device, scaler=None):
    model.train()

    running_loss = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for x, y, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y0 = remap_tensor_labels(y, LABEL_ORDER)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y0)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_n = x.size(0)
        running_loss += loss.item() * batch_n
        n_samples += batch_n

        pbar.set_postfix(loss=f"{running_loss / max(n_samples,1):.4f}")

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    n_samples = 0

    all_probs = []
    all_preds = []
    all_true = []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y0 = remap_tensor_labels(y, LABEL_ORDER)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y0)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_n = x.size(0)
        running_loss += loss.item() * batch_n
        n_samples += batch_n

        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_true.append(y0.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    metrics, per_class_df, report = compute_metrics(y_true, y_pred, prefix_name="Eval")

    try:
        ll = log_loss(y_true, y_prob, labels=np.arange(NUM_CLASSES))
    except Exception:
        ll = np.nan

    metrics["loss"] = float(running_loss / max(n_samples, 1))
    metrics["log_loss"] = None if np.isnan(ll) else float(ll)

    return metrics, per_class_df, report, y_true, y_pred, y_prob


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_model_once(
    model_name,
    params,
    train_dataset,
    val_dataset,
    fold_output_dir,
    random_state=SEED,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )

    y_train = subset_labels(train_dataset)
    class_weights = compute_class_weights(y_train, LABEL_ORDER).to(DEVICE)

    model = build_model(model_name, params)

    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer_name = params["optimizer"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, params["patience"] // 2),
        #verbose=True
    )

    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_score = -np.inf
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    history_rows = []

    n_epochs = params["epochs"]
    patience = params["patience"]

    print(f"Trainable params: {count_trainable_params(model):,}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss = run_one_epoch_train(model, train_loader, optimizer, criterion, DEVICE, scaler=scaler)
        val_metrics, _, _, _, _, _ = evaluate_loader(model, val_loader, criterion, DEVICE)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_log_loss": val_metrics["log_loss"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - t0,
        }
        history_rows.append(row)

        score_tuple = (
            val_metrics["f1_macro"],
            val_metrics["balanced_accuracy"],
            -(val_metrics["log_loss"] if val_metrics["log_loss"] is not None else 1e9),
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
        )

        scheduler.step(val_metrics["f1_macro"])

        if score_tuple > best_score if isinstance(best_score, tuple) else True:
            best_score = score_tuple
            best_epoch = epoch
            epochs_no_improve = 0

            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best_state = copy.deepcopy(state_dict)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(Path(fold_output_dir) / "train_history.csv", index=False)

    if best_state is None:
        raise RuntimeError("Training failed: no best model state was captured.")

    # reload best model
    final_model = build_model(model_name, params)
    final_model.load_state_dict(best_state)

    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        final_model = nn.DataParallel(final_model)

    final_model = final_model.to(DEVICE)

    val_metrics, val_per_class, val_report, y_val_true, y_val_pred, y_val_prob = evaluate_loader(
        final_model, val_loader, criterion, DEVICE
    )

    return {
        "model": final_model,
        "best_epoch": best_epoch,
        "history_df": history_df,
        "val_metrics": val_metrics,
        "val_per_class": val_per_class,
        "val_report": val_report,
        "y_val_true": y_val_true,
        "y_val_pred": y_val_pred,
        "y_val_prob": y_val_prob,
    }


def select_best_result(results_df):
    """
    Selection criterion:
    1) highest macro F1
    2) highest balanced accuracy
    3) lowest log-loss if available
    """
    df = results_df.copy()
    df["_log_loss_fill"] = df["log_loss_val"].fillna(np.inf)

    df = df.sort_values(
        by=["f1_macro_val", "balanced_acc_val", "_log_loss_fill"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    best_row = df.iloc[0].to_dict()
    df = df.drop(columns=["_log_loss_fill"])
    return best_row, df


def run_hyperparameter_search(model_name, param_grid, train_dataset, val_dataset, fold_output_dir):
    param_list = list(ParameterGrid(param_grid))
    print(f"Total hyperparameter combinations: {len(param_list)}")

    search_rows = []
    best_artifact = None
    best_tuple = None

    for i, params in enumerate(param_list, start=1):
        print("\n" + "-" * 100)
        print(f"[{i}/{len(param_list)}] Evaluating params:")
        print(json.dumps(params, indent=2))

        combo_dir = Path(fold_output_dir) / f"search_combo_{i:03d}"
        ensure_dir(combo_dir)

        fit_out = fit_model_once(
            model_name=model_name,
            params=params,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            fold_output_dir=combo_dir,
        )

        val_metrics = fit_out["val_metrics"]

        row = dict(params)
        row.update({
            "accuracy_val": val_metrics["accuracy"],
            "f1_weighted_val": val_metrics["f1_weighted"],
            "f1_macro_val": val_metrics["f1_macro"],
            "balanced_acc_val": val_metrics["balanced_accuracy"],
            "log_loss_val": np.nan if val_metrics["log_loss"] is None else val_metrics["log_loss"],
            "best_epoch": fit_out["best_epoch"],
        })
        search_rows.append(row)

        this_tuple = (
            row["f1_macro_val"],
            row["balanced_acc_val"],
            -(row["log_loss_val"] if not np.isnan(row["log_loss_val"]) else 1e9),
        )

        if best_tuple is None or this_tuple > best_tuple:
            best_tuple = this_tuple
            best_artifact = fit_out

        # free memory
        del fit_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(search_rows)
    best_row, sorted_df = select_best_result(results_df)

    results_df.to_csv(Path(fold_output_dir) / "hyperparameter_search_results.csv", index=False)
    sorted_df.to_csv(Path(fold_output_dir) / "hyperparameter_search_results_sorted.csv", index=False)

    return best_row, sorted_df


def fit_final_model_and_evaluate(
    model_name,
    best_params,
    train_dataset,
    val_dataset,
    test_dataset,
    fold_output_dir,
):
    trainval_dataset = ConcatDataset([train_dataset, val_dataset])

    trainval_loader = DataLoader(
        trainval_dataset,
        batch_size=best_params["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=best_params["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )

    # collect trainval labels
    y_train = subset_labels(train_dataset)
    y_val = subset_labels(val_dataset)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    class_weights = compute_class_weights(y_trainval, LABEL_ORDER).to(DEVICE)

    model = build_model(model_name, best_params)

    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if best_params["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=best_params["lr"],
            momentum=0.9,
            nesterov=True,
            weight_decay=best_params["weight_decay"],
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, best_params["patience"] // 2),
       # verbose=True
    )

    scaler = GradScaler("cuda",enabled=(DEVICE.type == "cuda"))
    best_state = None
    best_score = None
    best_epoch = -1
    epochs_no_improve = 0

    final_n_epochs = best_params["epochs"]
    final_patience = best_params["patience"]

    # small internal validation signal from trainval is not used here,
    # so we monitor train loss only for LR and keep best on train loss proxy.
    # Alternatively, one could keep the selected epoch from the first search.
    best_train_loss = np.inf

    history_rows = []

    for epoch in range(1, final_n_epochs + 1):
        train_loss = run_one_epoch_train(model, trainval_loader, optimizer, criterion, DEVICE, scaler=scaler)

        row = {
            "epoch": epoch,
            "trainval_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)

        scheduler.step(-train_loss)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            epochs_no_improve = 0
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best_state = copy.deepcopy(state_dict)
        else:
            epochs_no_improve += 1

        print(f"[Final fit epoch {epoch:03d}] trainval_loss={train_loss:.4f}")

        if epochs_no_improve >= final_patience:
            print(f"Final fit early stop at epoch {epoch} (best epoch: {best_epoch})")
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(Path(fold_output_dir) / "final_trainval_history.csv", index=False)

    if best_state is None:
        raise RuntimeError("No best state for final model.")

    final_model = build_model(model_name, best_params)
    final_model.load_state_dict(best_state)

    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        final_model = nn.DataParallel(final_model)
    final_model = final_model.to(DEVICE)

    test_metrics, test_per_class, test_report, y_test_true, y_test_pred, y_test_prob = evaluate_loader(
        final_model, test_loader, criterion, DEVICE
    )

    ckpt_path = Path(fold_output_dir) / "best_final_model.pt"
    state_to_save = final_model.module.state_dict() if isinstance(final_model, nn.DataParallel) else final_model.state_dict()
    torch.save(
        {
            "model_name": model_name,
            "best_params": best_params,
            "state_dict": state_to_save,
            "label_order": LABEL_ORDER,
            "class_names": CLASS_NAMES,
        },
        ckpt_path
    )

    return {
        "final_model": final_model,
        "test_metrics": test_metrics,
        "test_per_class": test_per_class,
        "test_report": test_report,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
        "checkpoint_path": str(ckpt_path),
    }


# =============================================================================
# PARAMETER GRIDS
# =============================================================================

PARAM_GRIDS = {
    "dino_cls": {
        "hidden_dim": [256, 512],
        "dropout": [0.2, 0.4],
        "freeze_backbone": [True, False],
        "optimizer": ["adamw"],
        "lr": [1e-4, 5e-4],
        "weight_decay": [1e-4, 1e-3],
        "batch_size": [batch_size],
        "epochs": [100],
        "patience": [5],
    },
    "resnet3d_18": {
        "dropout": [0.2, 0.4],
        "base_channels": [32],
        "optimizer": ["adamw"],
        "lr": [1e-4, 3e-4],
        "weight_decay": [1e-4, 1e-3],
        "batch_size": [batch_size],
        "epochs": [100],
        "patience": [5],
    },
}



# =============================================================================
# PER-FOLD PROCESSING
# =============================================================================

def process_single_fold(test_fold_dir, all_fold_dirs, output_root, model_name):
    fold_name = test_fold_dir.name
    fold_output_dir = Path(output_root) / model_name / fold_name
    ensure_dir(fold_output_dir)

    print("\n" + "=" * 100)
    print(f"Processing fold: {fold_name}")
    print(f"Model: {model_name}")
    print("=" * 100)

    train_fold_dirs = [p for p in all_fold_dirs if p != test_fold_dir]

    trainval_dataset_full = build_dataset_for_folders(train_fold_dirs, do_aug=False)
    train_dataset_full_aug = build_dataset_for_folders(train_fold_dirs, do_aug=True)
    test_dataset = build_dataset_for_folders([test_fold_dir], do_aug=False)

    # split on the non-augmented version, then map indices onto augmented version
    train_subset_plain, val_subset_plain = make_train_val_subsets(
        trainval_dataset_full, val_fraction=0.15, random_state=SEED
    )

    train_indices = train_subset_plain.indices
    val_indices = val_subset_plain.indices

    train_dataset = Subset(train_dataset_full_aug, train_indices)
    val_dataset = Subset(trainval_dataset_full, val_indices)

    best_row, results_df_sorted = run_hyperparameter_search(
        model_name=model_name,
        param_grid=PARAM_GRIDS[model_name],
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        fold_output_dir=fold_output_dir,
    )

    with open(Path(fold_output_dir) / "best_validation_result.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    best_params = {}
    for k, v in best_row.items():
        if k in PARAM_GRIDS[model_name]:
            best_params[k] = v

    eval_out = fit_final_model_and_evaluate(
        model_name=model_name,
        best_params=best_params,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        fold_output_dir=fold_output_dir,
    )

    # Save reports / confusion matrices
    test_metrics = eval_out["test_metrics"]
    test_per_class = eval_out["test_per_class"]
    test_report = eval_out["test_report"]

    test_per_class.to_csv(Path(fold_output_dir) / "test_per_class.csv", index=False)

    with open(Path(fold_output_dir) / "test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(test_report)
        f.write("\n")

    save_confusion_matrix(
        eval_out["y_test_true"], eval_out["y_test_pred"],
        Path(fold_output_dir) / "test_confusion_matrix_raw.svg",
        f"Test Confusion Matrix - {model_name} - {fold_name}",
        normalize=None,
        class_names=CLASS_NAMES
    )
    save_confusion_matrix(
        eval_out["y_test_true"], eval_out["y_test_pred"],
        Path(fold_output_dir) / "test_confusion_matrix_normalized.svg",
        f"Test Confusion Matrix - {model_name} - {fold_name} (normalized)",
        normalize="true",
        class_names=CLASS_NAMES
    )

    fold_summary = {
        "fold_name": fold_name,
        "model_name": model_name,
        "best_accuracy_val": float(best_row["accuracy_val"]),
        "best_f1_weighted_val": float(best_row["f1_weighted_val"]),
        "best_f1_macro_val": float(best_row["f1_macro_val"]),
        "best_balanced_acc_val": float(best_row["balanced_acc_val"]),
        "best_log_loss_val": None if pd.isna(best_row["log_loss_val"]) else float(best_row["log_loss_val"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision_weighted": float(test_metrics["precision_weighted"]),
        "test_recall_weighted": float(test_metrics["recall_weighted"]),
        "test_f1_weighted": float(test_metrics["f1_weighted"]),
        "test_precision_macro": float(test_metrics["precision_macro"]),
        "test_recall_macro": float(test_metrics["recall_macro"]),
        "test_f1_macro": float(test_metrics["f1_macro"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "checkpoint_path": eval_out["checkpoint_path"],
        "output_dir": str(fold_output_dir),
    }

    with open(Path(fold_output_dir) / "fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    print("\nSaved fold results to:", fold_output_dir)

    return {
        "fold_summary": fold_summary,
        "test_per_class_df": test_per_class.assign(fold_name=fold_name),
        "y_test_true": eval_out["y_test_true"],
        "y_test_pred": eval_out["y_test_pred"],
    }


# =============================================================================
# CROSS-VALIDATION AGGREGATION
# =============================================================================

def aggregate_cv_results(fold_outputs, output_root, model_name):
    model_output_dir = Path(output_root) / model_name
    ensure_dir(model_output_dir)

    fold_summary_df = pd.DataFrame([fo["fold_summary"] for fo in fold_outputs])
    fold_summary_csv = model_output_dir / "cross_validation_fold_results.csv"
    fold_summary_df.to_csv(fold_summary_csv, index=False)

    metric_cols = [
        "test_accuracy",
        "test_precision_weighted",
        "test_recall_weighted",
        "test_f1_weighted",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "test_balanced_accuracy"
    ]

    agg_rows = []
    for col in metric_cols:
        values = fold_summary_df[col].astype(float).values
        agg_rows.append({
            "metric": col,
            "mean": np.mean(values),
            "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
            "min": np.min(values),
            "max": np.max(values),
            "n_folds": len(values),
        })

    aggregate_metrics_df = pd.DataFrame(agg_rows)
    aggregate_metrics_csv = model_output_dir / "cross_validation_aggregate_metrics.csv"
    aggregate_metrics_df.to_csv(aggregate_metrics_csv, index=False)

    per_class_all = pd.concat([fo["test_per_class_df"] for fo in fold_outputs], ignore_index=True)
    per_class_summary = (
        per_class_all
        .groupby("class_name")[["precision", "recall", "f1_score", "support"]]
        .agg(["mean", "std", "min", "max"])
    )
    per_class_summary.columns = ["_".join(col).strip() for col in per_class_summary.columns.values]
    per_class_summary = per_class_summary.reset_index()

    per_class_all_csv = model_output_dir / "cross_validation_test_per_class_all_folds.csv"
    per_class_summary_csv = model_output_dir / "cross_validation_test_per_class_summary.csv"

    per_class_all.to_csv(per_class_all_csv, index=False)
    per_class_summary.to_csv(per_class_summary_csv, index=False)

    y_test_all = np.concatenate([fo["y_test_true"] for fo in fold_outputs], axis=0)
    y_pred_all = np.concatenate([fo["y_test_pred"] for fo in fold_outputs], axis=0)

    pooled_report_txt = classification_report(
        y_test_all,
        y_pred_all,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    pooled_report_file = model_output_dir / "pooled_test_classification_report.txt"
    with open(pooled_report_file, "w", encoding="utf-8") as f:
        f.write("Pooled test classification report across CV folds\n")
        f.write("=" * 100 + "\n\n")
        f.write(pooled_report_txt)
        f.write("\n")

    pooled_cm_raw = model_output_dir / "pooled_test_confusion_matrix_raw.svg"
    pooled_cm_norm = model_output_dir / "pooled_test_confusion_matrix_normalized.svg"

    save_confusion_matrix(
        y_test_all, y_pred_all,
        pooled_cm_raw,
        f"Pooled Test Confusion Matrix - {model_name} (raw)",
        normalize=None,
        class_names=CLASS_NAMES
    )
    save_confusion_matrix(
        y_test_all, y_pred_all,
        pooled_cm_norm,
        f"Pooled Test Confusion Matrix - {model_name} (normalized)",
        normalize="true",
        class_names=CLASS_NAMES
    )

    summary_txt = model_output_dir / "cross_validation_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Cross-validation summary for model: {model_name}\n")
        f.write("=" * 100 + "\n\n")
        f.write("Aggregate metrics across folds\n")
        f.write("-" * 80 + "\n")
        f.write(aggregate_metrics_df.to_string(index=False))
        f.write("\n\n")
        f.write("Per-fold test metrics\n")
        f.write("-" * 80 + "\n")
        f.write(fold_summary_df.to_string(index=False))
        f.write("\n\n")
        f.write("Per-class summary across folds\n")
        f.write("-" * 80 + "\n")
        f.write(per_class_summary.to_string(index=False))
        f.write("\n\n")
        f.write("Pooled test classification report\n")
        f.write("-" * 80 + "\n")
        f.write(pooled_report_txt)
        f.write("\n")

    print("\n" + "=" * 100)
    print(f"Cross-validation finished for model: {model_name}")
    print("Saved:")
    print(fold_summary_csv)
    print(aggregate_metrics_csv)
    print(per_class_all_csv)
    print(per_class_summary_csv)
    print(pooled_report_file)
    print(pooled_cm_raw)
    print(pooled_cm_norm)
    print(summary_txt)
    print("=" * 100)

    return {
        "fold_summary_df": fold_summary_df,
        "aggregate_metrics_df": aggregate_metrics_df,
        "per_class_summary_df": per_class_summary,
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_cross_validation_benchmark(patches_dir, output_root, model_name):
    all_fold_dirs = collect_fold_dirs(Path(patches_dir))
    ensure_dir(Path(output_root) / model_name)

    print("\nFound folds:")
    for f in all_fold_dirs:
        print(" -", f.name)

    fold_outputs = []
    for test_fold_dir in all_fold_dirs:
        fold_out = process_single_fold(
            test_fold_dir=test_fold_dir,
            all_fold_dirs=all_fold_dirs,
            output_root=output_root,
            model_name=model_name,
        )
        fold_outputs.append(fold_out)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return aggregate_cv_results(
        fold_outputs=fold_outputs,
        output_root=output_root,
        model_name=model_name,
    )


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    seed_everything(SEED)
    ensure_dir(output_root)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Run DINO + head
    run_cross_validation_benchmark(
        patches_dir=patches_dir,
        output_root=output_root,
        model_name="dino_cls"
    )

    # Run ResNet3D baseline
    run_cross_validation_benchmark(
        patches_dir=patches_dir,
        output_root=output_root,
        model_name="resnet3d_18"
    )