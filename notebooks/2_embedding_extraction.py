# -*- coding: utf-8 -*-
"""2_embedding_extraction.ipynb

"""
import sys
sys.path.append(".../Nuclei3DClassification/code/3DINO")

import os, glob, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff
import matplotlib.pyplot as plt
import skimage
from pathlib import Path
from tqdm import tqdm
from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config_3d

from scipy import ndimage, stats
from skimage import measure, filters, morphology, feature
from skimage.measure import marching_cubes, mesh_surface_area
from sklearn.decomposition import PCA

from dataset_helper import *

import time

#print("Pausing for 1 hour...")
#time.sleep(6000)  # 3600 seconds = 1 hour
#print("Resuming...")

#----------------------------------------------------------------
# Input
#----------------------------------------------------------------

extraded_features = "3DDINO"   # "3DDINO" or "Classic_features"

patches_dir  =  Path(".../Nuclei3DClassification/data/patches/")
out_dir_aug = Path(".../Nuclei3DClassification/data/embedding_aug_20000/")
target_per_label = 20000

save_embedings_raw = False
out_dir = Path(".../Nuclei3DClassification/data/embedding/")

save_augmented_examples = False
test_aug_dir = Path(".../Nuclei3DClassification/data/patches_test_aug/")


labels = [1,2,3,4,5]
target_DHW0 = (70,70,70)
target_DHW  = (112,112,112)

batch_size = 256
num_workers = 48
classic_batch_size = 48        # used only for handcrafted features
classic_mask_threshold = None  # None -> Otsu; or set a fixed value

SEED = 42

config_file = ".../Nuclei3DClassification/code/3DINO/dinov2/configs/train/vit3d_highres"
pretrained_weights = ".../Nuclei3DClassification/data/3dino_vit_weights.pth"


# Create ouput dir
test_aug_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
out_dir_aug.mkdir(parents=True, exist_ok=True)



def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

#----------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def extract_embeddings(
    dataset,
    model,
    save_path,
    batch_size,
    num_workers,
    device,
    worker_init_fn=None,
    generator=None,
    flatten_output=False,
    use_multi_gpu=True
):
    """
    Extract embeddings for all samples in a dataset and save them to disk.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset returning (x, y, path).
    model : torch.nn.Module
        Model used to extract embeddings.
    save_path : str
        Output .npz file path.
    batch_size : int
        Batch size for inference.
    num_workers : int
        Number of workers for DataLoader.
    device : torch.device
        Device for inference.
    worker_init_fn : callable, optional
        Worker init function for reproducibility.
    generator : torch.Generator, optional
        Generator for reproducibility.
    flatten_output : bool, default=False
        If True, flatten model outputs to shape (B, -1).
    use_multi_gpu : bool, default=False
        If True and multiple CUDA GPUs are available, wrap model with DataParallel.

    Returns
    -------
    embeddings : np.ndarray
        Extracted embeddings.
    labels : np.ndarray
        Corresponding labels.
    """

    # -----------------------------
    # Multi-GPU support (simple)
    # -----------------------------
    if use_multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    model.eval()

    all_emb = []
    all_y = []
    all_paths = []

    pbar = tqdm(total=len(dataset), desc="Extracting embeddings", unit="sample")

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)

            out = model(x)

            if not torch.is_tensor(out):
                raise TypeError(f"Expected model output to be a torch.Tensor, got {type(out)}")

            if out.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Output batch dimension {out.shape[0]} does not match input batch size {x.shape[0]}"
                )

            if flatten_output and out.ndim > 2:
                out = out.flatten(start_dim=1)

            all_emb.append(out.detach().cpu().numpy())

            if torch.is_tensor(y):
                all_y.append(y.detach().cpu().numpy())
            else:
                all_y.append(np.asarray(y))

            all_paths.extend(list(paths))

            # update by number of processed samples
            batch_n = x.size(0)
            pbar.update(batch_n)

            # show GPU memory
            if device.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                pbar.set_postfix({
                    "GPU_alloc_MB": f"{mem_alloc:.1f}",
                    "GPU_reserved_MB": f"{mem_reserved:.1f}"
                })

    pbar.close()

    if len(all_emb) == 0:
        raise ValueError("Dataset is empty. No embeddings were extracted.")

    embeddings = np.concatenate(all_emb, axis=0)
    labels = np.concatenate(all_y, axis=0)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(
        save_path,
        embeddings=embeddings,
        labels=labels,
        paths=np.asarray(all_paths, dtype=str)
    )

    print(f"Embeddings: {embeddings.shape} | Labels: {labels.shape} | Saved to: {save_path}")

    return embeddings, labels


# -----------------------------
# Classic handcrafted features
# -----------------------------

def _safe_div(a, b, eps=1e-8):
    return float(a) / (float(b) + eps)


def _central_slice_glcm_features(img2d, levels=32):
    """
    Compute a small set of Haralick-like GLCM texture features on one 2D slice.
    We later average across 3 orthogonal central slices.
    """
    img2d = np.asarray(img2d, dtype=np.float32)
    img2d = img2d - img2d.min()
    if img2d.max() > 0:
        img2d = img2d / img2d.max()
    img2d = np.clip((img2d * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)

    glcm = feature.graycomatrix(
        img2d,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=levels,
        symmetric=True,
        normed=True,
    )

    feats = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        vals = feature.graycoprops(glcm, prop)
        feats[f"haralick_{prop}"] = float(vals.mean())
    return feats


def _box_count(mask, box_size):
    """
    Number of occupied boxes of a given size.
    """
    Z, Y, X = mask.shape
    nz = int(np.ceil(Z / box_size))
    ny = int(np.ceil(Y / box_size))
    nx = int(np.ceil(X / box_size))

    pad_z = nz * box_size - Z
    pad_y = ny * box_size - Y
    pad_x = nx * box_size - X

    padded = np.pad(mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=False)
    reshaped = padded.reshape(nz, box_size, ny, box_size, nx, box_size)
    occupied = reshaped.any(axis=(1, 3, 5))
    return int(occupied.sum())


def _fractal_features(mask):
    """
    Approximate box-counting and Minkowski-Bouligand fractal dimensions.
    """
    sizes = [1, 2, 4, 8, 16]
    sizes = [s for s in sizes if s <= min(mask.shape)]
    if len(sizes) < 2:
        return {
            "fractal_box_counting": np.nan,
            "fractal_minkowski_bouligand": np.nan,
        }

    counts = np.array([_box_count(mask, s) for s in sizes], dtype=float)
    valid = counts > 0

    if valid.sum() < 2:
        box_fd = np.nan
    else:
        x = np.log(1.0 / np.array(sizes)[valid])
        y = np.log(counts[valid])
        box_fd = float(np.polyfit(x, y, 1)[0])

    # Minkowski-Bouligand approximation using dilated volumes
    radii = [1, 2, 3, 4]
    vols = []
    for r in radii:
        dil = ndimage.binary_dilation(mask, structure=morphology.ball(r))
        vols.append(dil.sum())
    vols = np.array(vols, dtype=float)

    valid = vols > 0
    if valid.sum() < 2:
        mb_fd = np.nan
    else:
        x = np.log(np.array(radii)[valid])
        y = np.log(vols[valid])
        slope = np.polyfit(x, y, 1)[0]
        # In 3D embedding space, MB dimension is approximated by the slope
        mb_fd = float(slope)

    return {
        "fractal_box_counting": box_fd,
        "fractal_minkowski_bouligand": mb_fd,
    }


def _weighted_lacunarity(mask, box_sizes=(1, 2, 3, 4, 5)):
    """
    Mean weighted lacunarity using box sizes 1..5 voxels.
    """
    vals = []
    weights = []

    mask = mask.astype(np.uint8)
    Z, Y, X = mask.shape

    for b in box_sizes:
        if b > min(mask.shape):
            continue

        nz = int(np.ceil(Z / b))
        ny = int(np.ceil(Y / b))
        nx = int(np.ceil(X / b))

        pad_z = nz * b - Z
        pad_y = ny * b - Y
        pad_x = nx * b - X

        padded = np.pad(mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        reshaped = padded.reshape(nz, b, ny, b, nx, b)
        masses = reshaped.sum(axis=(1, 3, 5)).astype(np.float32).ravel()

        if masses.size == 0 or masses.mean() == 0:
            continue

        lac = (np.mean(masses ** 2) / (np.mean(masses) ** 2 + 1e-8))
        vals.append(float(lac))
        weights.append(float(b))

    if len(vals) == 0:
        return np.nan

    vals = np.array(vals, dtype=float)
    weights = np.array(weights, dtype=float)
    return float(np.average(vals, weights=weights))


def _make_mask_from_volume(vol, threshold=None):
    """
    Build a binary object mask from a single-nucleus 3D patch.
    Assumption: foreground nucleus is the brightest coherent object.
    """
    vol = np.asarray(vol, dtype=np.float32)

    # If the patch is already effectively binary, use it directly.
    unique_vals = np.unique(vol)
    if unique_vals.size <= 3 and np.all(np.isin(unique_vals, [0, 1])):
        mask = vol > 0
    else:
        smooth = ndimage.gaussian_filter(vol, sigma=1.0)
        thr = filters.threshold_otsu(smooth) if threshold is None else threshold
        mask = smooth > thr

    mask = morphology.remove_small_objects(mask, min_size=16)
    mask = ndimage.binary_fill_holes(mask)

    # keep only largest connected component
    lab = measure.label(mask)
    props = measure.regionprops(lab)
    if len(props) == 0:
        return np.zeros_like(mask, dtype=bool)

    largest = max(props, key=lambda p: p.area).label
    return lab == largest


def _surface_voxels(mask):
    er = ndimage.binary_erosion(mask)
    return mask & (~er)


def _mean_surface_intensity_gradient(vol, mask):
    gz, gy, gx = np.gradient(vol.astype(np.float32))
    grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)
    surf = _surface_voxels(mask)
    if surf.sum() == 0:
        return np.nan
    return float(grad_mag[surf].mean())


def _mesh_curvature_approximations(mask, sigma=1.0):
    """
    Practical curvature approximations from a smoothed binary mask.
    This is robust and dependency-light.
    """
    smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)

    # marching cubes on smoothed mask
    try:
        verts, faces, normals, values = marching_cubes(smooth, level=0.5)
    except Exception:
        return {
            "curvature_mean": np.nan,
            "curvature_std": np.nan,
            "curvature_skewness": np.nan,
            "curvature_kurtosis": np.nan,
            "shape_index_mean": np.nan,
            "cvm": np.nan,
            "surface_area_mesh": np.nan,
        }

    area = float(mesh_surface_area(verts, faces))

    # Approximate curvature from divergence of normalized gradient field
    gz, gy, gx = np.gradient(smooth)
    mag = np.sqrt(gz**2 + gy**2 + gx**2) + 1e-8
    nz, ny, nx = gz / mag, gy / mag, gx / mag
    div_n = np.gradient(nz, axis=0) + np.gradient(ny, axis=1) + np.gradient(nx, axis=2)

    surf = np.logical_and(smooth > 0.25, smooth < 0.75)
    curv = div_n[surf]

    if curv.size < 5:
        return {
            "curvature_mean": np.nan,
            "curvature_std": np.nan,
            "curvature_skewness": np.nan,
            "curvature_kurtosis": np.nan,
            "shape_index_mean": np.nan,
            "cvm": np.nan,
            "surface_area_mesh": area,
        }

    # shape index approximation using normalized curvature scale
    curv_std = np.std(curv) + 1e-8
    shape_index = (2.0 / np.pi) * np.arctan(curv / curv_std)

    return {
        "curvature_mean": float(np.mean(curv)),
        "curvature_std": float(np.std(curv)),
        "curvature_skewness": float(stats.skew(curv, bias=False)),
        "curvature_kurtosis": float(stats.kurtosis(curv, fisher=True, bias=False)),
        "shape_index_mean": float(np.mean(shape_index)),
        "cvm": float(np.var(curv)),
        "surface_area_mesh": area,
    }


def extract_classic_features_from_volume(vol, threshold=None):
    """
    Extract classical morphology, intensity, texture, and fractal features
    from one 3D single-nucleus patch.

    Returns
    -------
    feature_dict : dict
    """
    vol = np.asarray(vol, dtype=np.float32)
    mask = _make_mask_from_volume(vol, threshold=threshold)

    if mask.sum() == 0:
        # consistent empty fallback
        feature_names = [
            "voxel_count", "surface_area_voxels", "surface_area_mesh",
            "elongation", "sphericity", "mean_radius", "radius_variance",
            "shape_index_mean", "surface_to_volume_ratio", "cvm",
            "curvature_mean", "curvature_std", "curvature_skewness", "curvature_kurtosis",
            "intensity_mean", "intensity_std", "intensity_skewness", "intensity_kurtosis",
            "fractal_box_counting", "fractal_minkowski_bouligand", "weighted_lacunarity_1_5",
            "mean_surface_intensity_gradient",
            "haralick_contrast", "haralick_dissimilarity", "haralick_homogeneity",
            "haralick_ASM", "haralick_energy", "haralick_correlation",
            "bbox_volume", "extent", "equivalent_diameter"
        ]
        return {k: np.nan for k in feature_names}

    coords = np.argwhere(mask)
    volume_vox = float(mask.sum())

    # surface voxels
    surf = _surface_voxels(mask)
    surface_area_vox = float(surf.sum())

    centroid = coords.mean(axis=0)
    dist = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
    mean_radius = float(dist.mean())
    radius_var = float(dist.var())

    # PCA-based elongation
    if coords.shape[0] >= 3:
        pca = PCA(n_components=3)
        pca.fit(coords)
        eigvals = np.maximum(pca.explained_variance_, 1e-8)
        eigvals = np.sort(eigvals)[::-1]
        elongation = float(np.sqrt(eigvals[0] / eigvals[-1]))
    else:
        elongation = np.nan

    # mesh-based geometry and curvature approximations
    curv_feats = _mesh_curvature_approximations(mask, sigma=1.0)
    surface_area_mesh = curv_feats["surface_area_mesh"]

    # sphericity: pi^(1/3) * (6V)^(2/3) / A
    sphericity = (np.pi ** (1.0 / 3.0)) * ((6.0 * volume_vox) ** (2.0 / 3.0))
    sphericity = _safe_div(sphericity, surface_area_mesh if np.isfinite(surface_area_mesh) else surface_area_vox)

    surface_to_volume_ratio = _safe_div(surface_area_mesh if np.isfinite(surface_area_mesh) else surface_area_vox, volume_vox)

    # intensity stats inside mask
    vals = vol[mask]
    intensity_mean = float(vals.mean())
    intensity_std = float(vals.std())
    intensity_skewness = float(stats.skew(vals, bias=False)) if vals.size > 3 else np.nan
    intensity_kurtosis = float(stats.kurtosis(vals, fisher=True, bias=False)) if vals.size > 3 else np.nan

    # fractals + lacunarity
    fract_feats = _fractal_features(mask)
    lac = _weighted_lacunarity(mask, box_sizes=(1, 2, 3, 4, 5))

    # surface intensity gradient
    msig = _mean_surface_intensity_gradient(vol, mask)

    # Haralick-like features from 3 orthogonal central slices
    cz, cy, cx = [s // 2 for s in vol.shape]
    glcm_feats_list = [
        _central_slice_glcm_features(vol[cz, :, :]),
        _central_slice_glcm_features(vol[:, cy, :]),
        _central_slice_glcm_features(vol[:, :, cx]),
    ]
    glcm_feats = {
        k: float(np.mean([d[k] for d in glcm_feats_list]))
        for k in glcm_feats_list[0].keys()
    }

    # some extra useful geometric features
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    bbox_volume = float((zmax - zmin + 1) * (ymax - ymin + 1) * (xmax - xmin + 1))
    extent = _safe_div(volume_vox, bbox_volume)
    equivalent_diameter = float((6.0 * volume_vox / np.pi) ** (1.0 / 3.0))

    feats = {
        "voxel_count": volume_vox,
        "surface_area_voxels": surface_area_vox,
        "surface_area_mesh": surface_area_mesh,
        "elongation": elongation,
        "sphericity": float(sphericity),
        "mean_radius": mean_radius,
        "radius_variance": radius_var,
        "shape_index_mean": curv_feats["shape_index_mean"],
        "surface_to_volume_ratio": surface_to_volume_ratio,
        "cvm": curv_feats["cvm"],
        "curvature_mean": curv_feats["curvature_mean"],
        "curvature_std": curv_feats["curvature_std"],
        "curvature_skewness": curv_feats["curvature_skewness"],
        "curvature_kurtosis": curv_feats["curvature_kurtosis"],
        "intensity_mean": intensity_mean,
        "intensity_std": intensity_std,
        "intensity_skewness": intensity_skewness,
        "intensity_kurtosis": intensity_kurtosis,
        "fractal_box_counting": fract_feats["fractal_box_counting"],
        "fractal_minkowski_bouligand": fract_feats["fractal_minkowski_bouligand"],
        "weighted_lacunarity_1_5": lac,
        "mean_surface_intensity_gradient": msig,
        **glcm_feats,
        "bbox_volume": bbox_volume,
        "extent": extent,
        "equivalent_diameter": equivalent_diameter,
    }

    return feats


def extract_classic_features(
    dataset,
    save_path,
    batch_size,
    num_workers,
    worker_init_fn=None,
    generator=None,
    threshold=None
):
    """
    Extract handcrafted features and save in the SAME format as embeddings:
    embeddings=<feature matrix>, labels=<labels>, paths=<paths>

    This preserves downstream compatibility.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    all_feat = []
    all_y = []
    all_paths = []
    feature_names = None

    pbar = tqdm(total=len(dataset), desc="Extracting classic features", unit="sample")

    for x, y, paths in loader:
        # x: (B, C, D, H, W) expected, with C=1
        x_np = x.detach().cpu().numpy()

        for i in range(x_np.shape[0]):
            vol = np.squeeze(x_np[i], axis=0)  # (D, H, W)
            feats = extract_classic_features_from_volume(vol, threshold=threshold)

            if feature_names is None:
                feature_names = list(feats.keys())

            all_feat.append([feats[k] for k in feature_names])

            if torch.is_tensor(y):
                all_y.append(y[i].item())
            else:
                all_y.append(y[i])

            all_paths.append(paths[i])
            pbar.update(1)

    pbar.close()

    if len(all_feat) == 0:
        raise ValueError("Dataset is empty. No classic features were extracted.")

    embeddings = np.asarray(all_feat, dtype=np.float32)
    labels = np.asarray(all_y)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(
        save_path,
        embeddings=embeddings,
        labels=labels,
        paths=np.asarray(all_paths, dtype=str),
        feature_names=np.asarray(feature_names, dtype=str)
    )

    print(f"Classic features: {embeddings.shape} | Labels: {labels.shape} | Saved to: {save_path}")
    return embeddings, labels, feature_names


# -----------------------------
## Test data augmentation
# -----------------------------


def save_aug_preview(dataset_noaug, dataset_aug, out_dir, n_examples=10, extra_name=""):
    """
    Save paired preview volumes from non-augmented and augmented datasets.

    Parameters
    ----------
    dataset_noaug : Dataset
        Dataset without augmentation.
    dataset_aug : Dataset
        Dataset with augmentation.
    out_dir : str or Path
        Output directory.
    n_examples : int, default=10
        Number of examples to save.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(dataset_noaug) != len(dataset_aug):
        raise ValueError(
            f"Dataset size mismatch: len(dataset_noaug)={len(dataset_noaug)} "
            f"!= len(dataset_aug)={len(dataset_aug)}"
        )

    picks = np.linspace(
        0,
        len(dataset_noaug) - 1,
        num=min(n_examples, len(dataset_noaug)),
        dtype=int
    )

    for i, idx in enumerate(picks):
        x0, y0, p0 = dataset_noaug[idx]
        x1, y1, p1 = dataset_aug[idx]

        if y0 != y1:
            raise ValueError(f"Label mismatch at idx={idx}: {y0} vs {y1}")

        if p0 != p1:
            raise ValueError(f"Path mismatch at idx={idx}: {p0} vs {p1}")

        v0 = x0.squeeze(0).numpy()  # (D, H, W)
        v1 = x1.squeeze(0).numpy()

        stem = Path(p0).stem

        tiff.imwrite(out_dir / f"noaug_{i:02d}_{extra_name}_{stem}_label{y0}.tif", v0)
        tiff.imwrite(out_dir / f"aug_{i:02d}_{extra_name}__{stem}_label{y0}.tif", v1)

    print(f"Saved {len(picks)} preview pairs in: {out_dir}")


# Calculations

if save_augmented_examples:
  # Get all subfolders
  subfolders = sorted([p for p in patches_dir.iterdir() if p.is_dir()])

  for img_folder in subfolders:
    print(f"\nProcessing folder: {img_folder.name}")

    # Create non-augmented dataset
    dataset_noaug = Tif3DDatasetSingle(
        base_dir=str(img_folder),
        labels=labels,
        target_dhw0=target_DHW0,
        target_dhw=target_DHW,
        do_aug=False,
        seed=SEED,
    )
    # Create augmented dataset
    dataset_aug = Tif3DDatasetSingle(
        base_dir=str(img_folder),
        labels=labels,
        target_dhw0=target_DHW0,
        target_dhw=target_DHW,
        do_aug=True,
        seed=SEED,
    )

    print(f"Found {len(dataset_noaug)} patches to augment -> {len(dataset_aug)} ")

    save_aug_preview(dataset_noaug, dataset_aug, test_aug_dir, n_examples=5, extra_name=img_folder.name)

# -----------------------------
## Get embedings
# -----------------------------

# -----------------------------
# Prepare extractor
# -----------------------------

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if extraded_features == "3DDINO":
    cfg = load_and_merge_config_3d(config_file)
    model = build_model_for_eval(cfg, pretrained_weights)
    model = model.to(device)
    model.eval()
    print("DINO3D model ready at:", device)

elif extraded_features == "Classic_features":
    print("Using handcrafted classic features extractor (CPU/GPU independent).")

else:
    raise ValueError(
        f"Unknown extraded_features='{extraded_features}'. "
        f"Use '3DDINO' or 'Classic_features'."
    )

g = torch.Generator()
g.manual_seed(SEED)

# Calculations of embedings

# Get all subfolders
subfolders = sorted([p for p in patches_dir.iterdir() if p.is_dir()])

for img_folder in subfolders:
    print(f"\nProcessing folder: {img_folder.name}")

    # Create non-augmented dataset
    dataset_noaug = Tif3DDatasetSingle(
        base_dir=str(img_folder),
        labels=labels,
        target_dhw0=target_DHW0,
        target_dhw=target_DHW,
        do_aug=False,
        seed=SEED,
    )

    # Create augmented dataset
    dataset_aug = Tif3DDatasetSingle(
        base_dir=str(img_folder),
        labels=labels,
        target_dhw0=target_DHW0,
        target_dhw=target_DHW,
        do_aug=True,
        target_per_label=target_per_label,
        seed=SEED,
    )

    print(f"Found {len(dataset_noaug)} original patches and {len(dataset_aug)} augmented ")

    if save_embedings_raw:
      save_path = out_dir / f"{img_folder.name}.npz"

      if extraded_features == "3DDINO":
          embeddings, labels_emb = extract_embeddings(
              dataset_noaug,
              model,
              save_path,
              batch_size,
              num_workers,
              device,
              worker_init_fn=seed_worker,
              generator=g
          )
      else:
          embeddings, labels_emb, feature_names = extract_classic_features(
              dataset_noaug,
              save_path,
              classic_batch_size,
              num_workers,
              worker_init_fn=seed_worker,
              generator=g,
              threshold=classic_mask_threshold
          )

      print(f"Saved at: {save_path}")

    save_path_aug = out_dir_aug / f"{img_folder.name}_{target_per_label}_samples_aug.npz"

    if extraded_features == "3DDINO":
        embeddings_aug, labels_emb_aug = extract_embeddings(
            dataset_aug,
            model,
            save_path_aug,
            batch_size,
            num_workers,
            device,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        embeddings_aug, labels_emb_aug, feature_names_aug = extract_classic_features(
            dataset_aug,
            save_path_aug,
            classic_batch_size,
            num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            threshold=classic_mask_threshold
        )

    print(f"Saved at: {save_path_aug}")

