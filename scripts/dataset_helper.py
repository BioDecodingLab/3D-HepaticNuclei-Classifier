"""Image normalization, spatial standardization and mask-aware datasets."""

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from pathlib import Path
from skimage.transform import resize
from torch.utils.data import Dataset
from pipeline_common import stable_seed


def normalize_image(vol, minv=-1.0, maxv=1.0):
    """
    Per-volume min–max normalization.

    Parameters
    ----------
    vol : ndarray
        Input volume
    minv : float
        Output minimum
    maxv : float
        Output maximum

    Returns
    -------
    ndarray
        Normalized volume
    """

    vol = vol.astype(np.float32, copy=False)

    lo = np.amin(vol)
    hi = np.amax(vol)

    if hi <= lo:
        return np.full_like(vol, minv, dtype=np.float32)

    # normalize to [0,1]
    vol = (vol - lo) / (hi - lo)

    # clip
    vol = np.clip(vol, 0.0, 1.0)

    # rescale to [minv,maxv]
    vol = vol * (maxv - minv) + minv

    return vol


def center_pad_3d(vol, target_dhw, pad_value=0, only_pad=True, verbose=False):
    """
    Center pad or crop a 3D volume.

    Parameters
    ----------
    vol : np.ndarray
        Input volume of shape (D, H, W).
    target_dhw : tuple of int
        Target shape (tD, tH, tW).
        Minimum padded shape; oversized inputs use a cube of their maximum dimension.
    pad_value : float, optional
        Constant value used for padding.
    only_pad : bool, optional
        If False:
            Center pad/crop to exactly target_dhw.
        If True:
            Do not crop. Instead, pad to a centered cube whose size is the
            largest input dimension when it exceeds the requested target.
    verbose : bool, optional
        If True, print debug information.

    Returns
    -------
    np.ndarray
        Output 3D volume.
        - shape = target_dhw if only_pad=False
        - with only_pad=True, use the target or a larger enclosing cube
    """
    if not isinstance(vol, np.ndarray):
        raise TypeError(f"`vol` must be a numpy array, got {type(vol)}")

    if vol.ndim != 3:
        raise ValueError(f"`vol` must be 3D with shape (D,H,W), got {vol.shape}")

    if len(target_dhw) != 3:
        raise ValueError(f"`target_dhw` must have length 3, got {target_dhw}")

    if not all(isinstance(x, (int, np.integer)) for x in target_dhw):
        raise TypeError(
            f"All entries in `target_dhw` must be integers, got {target_dhw}"
        )

    if min(target_dhw) <= 0:
        raise ValueError(f"All target dimensions must be positive, got {target_dhw}")

    D, H, W = vol.shape

    # verbose = False
    # if only_pad and max(D, H, W) > np.amax(target_dhw) and verbose0:
    #    verbose = True

    if verbose:
        print(f"Input shape: {vol.shape}")
        print(f"Requested target_dhw: {target_dhw}")
        print(f"only_pad: {only_pad}")

    if only_pad and max(D, H, W) > np.amax(target_dhw):
        # pad to cube of largest input dimension, no cropping
        cube_size = max(D, H, W)
        tD = tH = tW = cube_size

    else:
        tD, tH, tW = target_dhw

    # compute padding
    pad_d = max(0, tD - D)
    pad_h = max(0, tH - H)
    pad_w = max(0, tW - W)

    pad_width = (
        (pad_d // 2, pad_d - pad_d // 2),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
    )

    if verbose:
        print(f"Padding: D={pad_width[0]}, H={pad_width[1]}, W={pad_width[2]}")

    if pad_d or pad_h or pad_w:
        vol = np.pad(
            vol,
            pad_width=pad_width,
            mode="constant",
            constant_values=pad_value,
        )

    if only_pad:
        if verbose:
            print(f"Output shape (only_pad=True): {vol.shape}")
        return vol

    # center crop if needed
    Dp, Hp, Wp = vol.shape
    sD = (Dp - tD) // 2
    sH = (Hp - tH) // 2
    sW = (Wp - tW) // 2

    if verbose:
        print(f"Shape after padding: {vol.shape}")
        print(f"Crop starts: D={sD}, H={sH}, W={sW}")

    out = vol[sD : sD + tD, sH : sH + tH, sW : sW + tW]

    if verbose:
        print(f"Output shape: {out.shape}")

    return out


def random_rot90_3d(vol, rng: np.random.RandomState, p=0.5):
    """
    Apply a random 3D rotation from the 24 cube symmetries (no interpolation).

    Parameters
    ----------
    vol : np.ndarray
        Input volume (D, H, W)
    rng : np.random.RandomState
        Random generator

    Returns
    -------
    np.ndarray
        Rotated volume
    """

    out = vol.copy()

    # all 24 possible rotations generated via axis permutations + flips
    axes_permutations = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]

    perm = axes_permutations[rng.randint(len(axes_permutations))]
    out = np.transpose(out, perm)

    # random flips
    if rng.rand() < p:
        out = np.flip(out, axis=0)
    if rng.rand() < p:
        out = np.flip(out, axis=1)
    if rng.rand() < p:
        out = np.flip(out, axis=2)

    return np.ascontiguousarray(out)


def _gaussian_1d_kernel(sigma: float, radius: int):
    """
    Generate a normalized 1D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    radius : int
        Kernel radius (kernel size = 2*radius + 1).

    Returns
    -------
    torch.Tensor
        1D normalized Gaussian kernel.
    """
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma * sigma))
    k = k / (k.sum() + 1e-8)
    return k


def gaussian_blur_3d(
    vol, rng: np.random.RandomState, p=0.5, sigma_range=(0.5, 2.0), mask=None
):
    """
    Apply random Gaussian blur to a 3D volume using separable convolution.

    The blur is applied with probability `p`. The Gaussian kernel width
    is sampled randomly from `sigma_range`. Background voxels (value 0)
    are preserved and restored after the blur.

    Parameters
    ----------
    vol : ndarray
        Input volume with shape (D, H, W) and values in [0,1].
    rng : np.random.RandomState
        Random number generator.
    p : float
        Probability of applying the blur.
    sigma_range : tuple
        Range of sigma values used to generate the Gaussian kernel.

    Returns
    -------
    ndarray
        Blurred volume with the same shape as the input.
    """

    out = vol.copy()

    if rng.rand() < p:

        sigma = float(rng.uniform(*sigma_range))
        radius = int(max(1, round(3 * sigma)))

        k1 = _gaussian_1d_kernel(sigma, radius)

        # Convert volume to tensor (1,1,D,H,W)
        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(torch.float32)
        k1 = k1.to(dtype=x.dtype, device=x.device)

        # Separable kernels for each axis
        kW = k1.view(1, 1, 1, 1, -1)
        kH = k1.view(1, 1, 1, -1, 1)
        kD = k1.view(1, 1, -1, 1, 1)

        # Blur along width
        x = F.pad(x, (radius, radius, 0, 0, 0, 0), mode="reflect")
        x = F.conv3d(x, kW)

        # Blur along height
        x = F.pad(x, (0, 0, radius, radius, 0, 0), mode="reflect")
        x = F.conv3d(x, kH)

        # Blur along depth
        x = F.pad(x, (0, 0, 0, 0, radius, radius), mode="reflect")
        x = F.conv3d(x, kD)

        out = x.squeeze(0).squeeze(0).cpu().numpy()

        # Restore background
        out[(vol == 0) if mask is None else ~mask] = 0

        # Ensure valid range
        out = np.clip(out, 0, 1)

    return out.astype(np.float32)


def standardize_pair(image, mask, initial=(70, 70, 70), target=(112, 112, 112)):
    """Keep the original image pad/resize calls; carry the original mask separately."""
    if image.ndim != 3 or image.shape != mask.shape or not mask.any():
        raise ValueError("Expected aligned, nonempty 3D intensity/mask pair")
    image = normalize_image(image, 0.0, 1.0)
    image = center_pad_3d(image, initial, only_pad=True)
    mask = center_pad_3d(mask.astype(bool), initial, only_pad=True)
    image = resize(
        image, target, order=0, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    mask = resize(
        mask.astype(np.uint8), target, order=0, preserve_range=True, anti_aliasing=False
    ).astype(bool)
    image[~mask] = 0
    if not mask.any():
        raise ValueError("Mask disappeared during resize")
    return image, mask


def augment_pair(image, mask, seed, p=0.3):
    """Original probabilities/strengths; geometry shared, photometry mask-preserving."""
    rng = np.random.RandomState(seed)
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    perm = perms[rng.randint(6)]
    image, mask = image.transpose(perm).copy(), mask.transpose(perm).copy()
    for axis in range(3):
        if rng.rand() < p:
            image, mask = np.flip(image, axis).copy(), np.flip(mask, axis).copy()
    if rng.rand() < p:
        image = image * (1 + rng.uniform(-0.1, 0.1)) + rng.uniform(-0.05, 0.05)
    image = np.clip(image, 0, 1)
    image[~mask] = 0
    # Separable reflect-padded Gaussian blur; preserve the independently held mask.
    image = gaussian_blur_3d(image, rng, p, sigma_range=(0.4, 1.2), mask=mask)
    if rng.rand() < p:
        image += rng.normal(0, 0.1, image.shape).astype(np.float32)
    image = np.clip(image, 0, 1).astype(np.float32)
    image[~mask] = 0
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def load_pair(root, record, initial=(70, 70, 70), target=(112, 112, 112)):
    image = tifffile.imread(Path(root) / record["image_path"])
    mask = tifffile.imread(Path(root) / record["mask_path"]).astype(bool)
    return standardize_pair(image, mask, initial, target)


class Tif3DDatasetSingle(Dataset):
    """Manifest rows, not directory discovery. No internal splitting/resampling."""

    def __init__(
        self,
        root,
        records,
        initial=(70, 70, 70),
        target=(112, 112, 112),
        do_aug=False,
        seed=42,
        epoch_augmentation=False,
        return_mask=False,
    ):
        self.root, self.records = Path(root), list(records)
        self.initial, self.target = tuple(initial), tuple(target)
        self.do_aug, self.seed = do_aug, seed
        self.epoch_augmentation, self.return_mask = epoch_augmentation, return_mask
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        record = self.records[idx]
        image, mask = load_pair(self.root, record, self.initial, self.target)
        if self.do_aug:
            seed = int(
                record.get(
                    "augmentation_seed", stable_seed(self.seed, record["sample_id"])
                )
            )
            if self.epoch_augmentation:
                seed = stable_seed(seed, self.epoch)
            image, mask = augment_pair(image, mask, seed)
        x = torch.from_numpy(image * 2 - 1).unsqueeze(0)
        if self.return_mask:
            return x, torch.from_numpy(mask), int(record["label"]), record["sample_id"]
        return x, int(record["label"]) - 1, record["sample_id"]
