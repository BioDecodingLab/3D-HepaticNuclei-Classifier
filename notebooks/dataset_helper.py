
import os, glob, random
import numpy as np
import skimage
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


#----------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------


def normalize_image(vol, minv=-1.0, maxv=1.0):
    """
    Percentile normalization of a 3D volume.

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
        Used only when only_pad=False.
    pad_value : float, optional
        Constant value used for padding.
    only_pad : bool, optional
        If False:
            Center pad/crop to exactly target_dhw.
        If True:
            Do not crop. Instead, pad to a centered cube whose size is the
            largest dimension of the input volume. if it is larget tahn teh dime of the target
    verbose : bool, optional
        If True, print debug information.

    Returns
    -------
    np.ndarray
        Output 3D volume.
        - shape = target_dhw if only_pad=False
        - shape = (M, M, M) with M=max(D,H,W) if only_pad=True
    """
    if not isinstance(vol, np.ndarray):
        raise TypeError(f"`vol` must be a numpy array, got {type(vol)}")

    if vol.ndim != 3:
        raise ValueError(f"`vol` must be 3D with shape (D,H,W), got {vol.shape}")

    if len(target_dhw) != 3:
        raise ValueError(f"`target_dhw` must have length 3, got {target_dhw}")

    if not all(isinstance(x, (int, np.integer)) for x in target_dhw):
        raise TypeError(f"All entries in `target_dhw` must be integers, got {target_dhw}")

    if min(target_dhw) <= 0:
        raise ValueError(f"All target dimensions must be positive, got {target_dhw}")

    D, H, W = vol.shape

    #verbose = False
    #if only_pad and max(D, H, W) > np.amax(target_dhw) and verbose0:
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

    out = vol[sD:sD + tD, sH:sH + tH, sW:sW + tW]

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
        (0,1,2),(0,2,1),
        (1,0,2),(1,2,0),
        (2,0,1),(2,1,0)
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

def intensity_aug(vol, rng: np.random.RandomState, p=0.5):
    """
    Random intensity augmentation for a 3D volume.

    Applies random scale and shift while keeping background at 0.
    """

    out = vol.copy()

    if rng.rand() < p:
        scale = 1.0 + rng.uniform(-0.2, 0.2)
        shift = rng.uniform(-0.1, 0.1)
        out = out * scale + shift

    # clip always
    out = np.clip(out, 0.0, 1.0)

    # restore background
    out[vol == 0] = 0

    return out

def gaussian_noise(vol, rng: np.random.RandomState, p=0.5, sigma=0.1):
    """
    Add Gaussian noise to a volume while keeping background at 0.
    """

    out = vol.copy()

    if rng.rand() < p:
        noise = rng.normal(0, sigma, size=vol.shape).astype(np.float32)
        out = out + noise

    out = np.clip(out, 0.0, 1.0)

    # restore background
    out[vol == 0] = 0

    return out


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


def gaussian_blur_3d(vol, rng: np.random.RandomState, p=0.5, sigma_range=(0.5, 2.0)):
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
        out[vol == 0] = 0

        # Ensure valid range
        out = np.clip(out, 0, 1)

    return out.astype(np.float32)




def augment_3dimage(vol, rng: np.random.RandomState, p=0.5):
    """
    Apply a sequence of random augmentations to a 3D image.

    The pipeline includes:
    - random 3D rotation
    - random intensity augmentation 
    - random Gaussian blur
    - random Gaussian noise

    Parameters
    ----------
    vol : ndarray
        Input volume with shape (D, H, W).
    rng : np.random.RandomState
        Random number generator.
    p : float
        Probability used for each stochastic augmentation step.

    Returns
    -------
    ndarray
        Augmented volume with the same shape as the input.
    """

    # Geometric augmentation
    vol = random_rot90_3d(vol, rng, p)

    # Photometric augmentation
    vol = intensity_aug(vol, rng, p)

    # Blur + noise
    vol = gaussian_blur_3d(vol, rng, p, sigma_range=(0.4, 1.2))
    vol = gaussian_noise(vol, rng, p, sigma=0.1)

    return vol



class Tif3DDatasetSingle(Dataset):
    def __init__(
        self,
        base_dir,
        labels,
        target_dhw0,
        target_dhw,
        target_per_label=None,
        do_aug=False,
        seed=42
    ):
        self.items = []
        self.target_dhw = target_dhw
        self.target_dhw0 = target_dhw0
        self.target_per_label = target_per_label
        self.do_aug = do_aug
        self.seed = seed

        # Temporary storage of original items by label
        items_by_label = {lv: [] for lv in labels}

        for lv in labels:
            label_folder = os.path.join(base_dir, f"label_{lv}")

            if not os.path.isdir(label_folder):
                print(f"Warning: folder not found: {label_folder}")
                continue

            files = sorted(
                glob.glob(os.path.join(label_folder, "*.tif")) +
                glob.glob(os.path.join(label_folder, "*.tiff"))
            )

            if len(files) == 0:
                print(f"Warning: no TIFF files found in {label_folder}")

            items_by_label[lv].extend([(f, lv) for f in files])

        # Check if dataset is empty
        n_total = sum(len(v) for v in items_by_label.values())
        if n_total == 0:
            raise ValueError(f"No TIFF files found in dataset: {base_dir}")

        # If target_per_label is given, resample with replacement
        if target_per_label is not None:
            rng = random.Random(seed)

            for lv in labels:
                src = items_by_label[lv]

                if len(src) == 0:
                    raise ValueError(f"No TIFF files found for label_{lv} in {base_dir}")

                for _ in range(target_per_label):
                    self.items.append(rng.choice(src))

            rng.shuffle(self.items)

        else:
            # Keep original dataset
            for lv in labels:
                self.items.extend(items_by_label[lv])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]

        # Load volume, expected shape (D, H, W)
        vol = tiff.imread(path).astype(np.float32)

        # Initial normalization to [0, 1]
        vol = normalize_image(vol, minv=0.0, maxv=1.0)

        # Pad then resize
        vol = center_pad_3d(vol, self.target_dhw0, only_pad=True)
        mask = vol != 0

        vol = skimage.transform.resize(
            vol,
            self.target_dhw,
            order=3,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)

        mask_resized = skimage.transform.resize(
            mask.astype(np.uint8),
            self.target_dhw,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(bool)

        vol[~mask_resized] = 0


        # Augmentation in [0,1]
        if self.do_aug:
            rng = np.random.RandomState(self.seed + idx)
            vol = augment_3dimage(vol, rng, p=0.5)

        # Final clipping in [0,1]
        vol = np.clip(vol, 0.0, 1.0).astype(np.float32, copy=False)

        # Convert from [0,1] to [-1,1] as model expects it
        vol = (vol * 2.0) - 1.0

        # Convert to tensor: (1, D, H, W)
        x = torch.from_numpy(vol).unsqueeze(0)

        return x, int(y), path

