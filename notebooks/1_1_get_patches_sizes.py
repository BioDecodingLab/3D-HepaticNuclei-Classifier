import os
from pathlib import Path
import argparse
from collections import Counter

import tifffile
import matplotlib.pyplot as plt
import numpy as np


def find_tiff_files(root_dir):
    """Recursively find all TIFF files in root_dir."""
    root = Path(root_dir)
    tiff_extensions = {".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in tiff_extensions]


def get_image_size_and_volume(tiff_path):
    """
    Return width, height, and volume of a TIFF image.

    Assumes:
    - 2D image: (H, W)          -> volume = H * W
    - 3D image: (..., H, W)     -> volume = product of all dimensions
    - multi-channel image: uses full shape product as volume
    """
    try:
        img = tifffile.imread(tiff_path)
        if img.ndim < 2:
            raise ValueError(f"Image has invalid shape: {img.shape}")

        height = img.shape[-2]
        width = img.shape[-1]
        volume = int(np.prod(img.shape))

        return width, height, np.log10(volume)
    except Exception as e:
        print(f"Could not read {tiff_path}: {e}")
        return None


def analyze_tiff_sizes(root_dir, output_png="tiff_size_histograms.png"):
    tiff_files = find_tiff_files(root_dir)

    if not tiff_files:
        print("No TIFF files found.")
        return

    widths = []
    heights = []
    volumes = []
    valid_files = []
    subfolder_counts = Counter()

    for tiff_file in tiff_files:
        size = get_image_size_and_volume(tiff_path=tiff_file)
        if size is not None:
            width, height, volume = size
            widths.append(width)
            heights.append(height)
            volumes.append(volume)
            valid_files.append(tiff_file)

            rel_parent = str(tiff_file.parent.relative_to(root_dir))
            subfolder_counts[rel_parent] += 1

    if not widths or not heights or not volumes:
        print("No valid TIFF images could be read.")
        return

    min_width = min(widths)
    max_width = max(widths)
    mean_width = np.mean(widths)

    min_height = min(heights)
    max_height = max(heights)
    mean_height = np.mean(heights)

    min_volume = min(volumes)
    max_volume = max(volumes)
    mean_volume = np.mean(volumes)

    print(f"Number of TIFF images found: {len(tiff_files)}")
    print(f"Number of valid TIFF images read: {len(valid_files)}")
    print()
    print(f"Width   -> min: {min_width}, mean: {mean_width:.2f}, max: {max_width}")
    print(f"Height  -> min: {min_height}, mean: {mean_height:.2f}, max: {max_height}")
    print(f"Volume  -> min: {min_volume}, mean: {mean_volume:.2f}, max: {max_volume}")
    print()

    print("Number of images per subfolder:")
    for subfolder, count in sorted(subfolder_counts.items()):
        print(f"  {subfolder}: {count}")

    # Create histograms
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.hist(widths, bins=30, edgecolor="black")
    plt.axvline(mean_width, linestyle="--", linewidth=2, label=f"Mean = {mean_width:.2f}")
    plt.axvline(max_width, linestyle=":", linewidth=2, label=f"Max = {max_width}")
    plt.axvline(min_width, linestyle=":", linewidth=2, label=f"Min = {min_width}")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Number of images")
    plt.title("Distribution of image widths")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(heights, bins=30, edgecolor="black")
    plt.axvline(mean_height, linestyle="--", linewidth=2, label=f"Mean = {mean_height:.2f}")
    plt.axvline(max_height, linestyle=":", linewidth=2, label=f"Max = {max_height}")
    plt.axvline(min_height, linestyle=":", linewidth=2, label=f"Min = {min_height}")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Number of images")
    plt.title("Distribution of image heights")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(volumes, bins=30, edgecolor="black")
    plt.axvline(mean_volume, linestyle="--", linewidth=2, label=f"Mean = {mean_volume:.2f}")
    plt.axvline(max_volume, linestyle=":", linewidth=2, label=f"Max = {max_volume}")
    plt.axvline(min_volume, linestyle=":", linewidth=2, label=f"Min = {min_volume}")
    plt.xlabel("Volume (pixels³ or total pixels)")
    plt.ylabel("Number of images")
    plt.title("Distribution of image volumes (log10)")
    plt.legend()

    plt.tight_layout()
    #plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nHistogram saved as: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze TIFF image sizes in a folder and its subfolders."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the top-level directory containing TIFF files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tiff_size_histograms.png",
        help="Output PNG file for the histograms"
    )

    args = parser.parse_args()
    analyze_tiff_sizes(args.root_dir, args.output)