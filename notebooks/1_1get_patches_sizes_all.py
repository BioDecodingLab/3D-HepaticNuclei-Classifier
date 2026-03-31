import argparse
from pathlib import Path
from collections import Counter, defaultdict

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
    - 2D image: (H, W)       -> volume = H * W
    - 3D image: (..., H, W)  -> volume = product of all dimensions
    - multi-channel image    -> uses full shape product as volume
    """
    try:
        img = tifffile.imread(tiff_path)

        if img.ndim < 2:
            raise ValueError(f"Image has invalid shape: {img.shape}")

        height = img.shape[-2]
        width = img.shape[-1]
        volume = int(np.prod(img.shape))

        return width, height, volume

    except Exception as e:
        print(f"Could not read {tiff_path}: {e}")
        return None


def safe_name(text):
    """Convert class/folder names into safe file names."""
    return str(text).replace("\\", "_").replace("/", "_").replace(" ", "_")


def save_histogram(data, mean_value, min_value, max_value, title, xlabel, output_path):
    plt.figure(figsize=(10, 6))  
    plt.hist(data, bins=30, edgecolor="black")
    plt.axvline(mean_value, linestyle="--", linewidth=2, label=f"Mean = {mean_value:.2f}")
    plt.axvline(min_value, linestyle=":", linewidth=2, label=f"Min = {min_value}")
    plt.axvline(max_value, linestyle=":", linewidth=2, label=f"Max = {max_value}")
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    #plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.savefig(output_path, bbox_inches="tight") 
    plt.close()


def analyze_tiff_sizes(root_dir, output_dir="tiff_histograms"):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = find_tiff_files(root_dir)

    if not tiff_files:
        print("No TIFF files found.")
        return

    widths = []
    heights = []
    volumes_by_class = defaultdict(list)

    valid_files = []
    subfolder_counts = Counter()

    for tiff_file in tiff_files:
        size = get_image_size_and_volume(tiff_file)

        if size is not None:
            width, height, volume = size

            widths.append(width)
            heights.append(height)

            clase = str(tiff_file.parent.relative_to(root_dir))
            volumes_by_class[clase].append(volume)

            valid_files.append(tiff_file)
            subfolder_counts[clase] += 1

    if not widths or not heights:
        print("No valid TIFF images could be read.")
        return

    # Width global
    min_width = min(widths)
    max_width = max(widths)
    mean_width = np.mean(widths)

    # Height global
    min_height = min(heights)
    max_height = max(heights)
    mean_height = np.mean(heights)

    print(f"Number of TIFF images found: {len(tiff_files)}")
    print(f"Number of valid TIFF images read: {len(valid_files)}")
    print()

    print(f"Width (global)  -> min: {min_width}, mean: {mean_width:.2f}, max: {max_width}")
    print(f"Height (global) -> min: {min_height}, mean: {mean_height:.2f}, max: {max_height}")
    print()

    print("Number of images per class:")
    for clase, count in sorted(subfolder_counts.items()):
        print(f"  {clase}: {count}")

    print("\nVolume statistics by class:")
    for clase in sorted(volumes_by_class.keys()):
        volumes = volumes_by_class[clase]
        min_volume = min(volumes)
        max_volume = max(volumes)
        mean_volume = np.mean(volumes)

        print(f"  {clase} -> min: {min_volume}, mean: {mean_volume:.2f}, max: {max_volume}")

    # Histogram
    save_histogram(
        data=widths,
        mean_value=mean_width,
        min_value=min_width,
        max_value=max_width,
        title="Width (Global)",
        xlabel="Width (pixels)",
        output_path=output_dir / "width_global.svg"
    )

    save_histogram(
        data=heights,
        mean_value=mean_height,
        min_value=min_height,
        max_value=max_height,
        title="Height (Global)",
        xlabel="Height (pixels)",
        output_path=output_dir / "height_global.svg"
    )

    for clase in sorted(volumes_by_class.keys()):
        volumes = volumes_by_class[clase]
        min_volume = min(volumes)
        max_volume = max(volumes)
        mean_volume = np.mean(volumes)

        class_filename = f"volume_{safe_name(clase)}.svg"

        save_histogram(
            data=volumes,
            mean_value=mean_volume,
            min_value=min_volume,
            max_value=max_volume,
            title=f"Volume - {clase}",
            xlabel="Volume (total pixels)",
            output_path=output_dir / class_filename
        )

    print(f"\nSeparate histograms saved in: {output_dir.resolve()}")


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
        "--output_dir",
        type=str,
        default="tiff_histograms",
        help="Folder where separate histogram SVG files will be saved"
    )

    args = parser.parse_args()
    analyze_tiff_sizes(args.root_dir, args.output_dir)