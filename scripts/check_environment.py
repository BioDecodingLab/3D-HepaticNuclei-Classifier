#!/usr/bin/env python3
"""Check installed dependencies, external 3DINO imports and optional GPU inference."""

import argparse
import importlib
from importlib.metadata import version
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    for name in ["dino-repo", "dino-config", "dino-weights"]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument(
        "--forward-check",
        action="store_true",
        help="Load the external checkpoint and test one synthetic 112-cube",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Use Python 3.12 in nuclei_classification; found {sys.version}"
        )
    print("Python:", sys.executable, platform.python_version(), flush=True)
    # Import compiled packages too: package metadata alone cannot detect ABI errors.
    for module, distribution in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("skimage", "scikit-image"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("tifffile", "tifffile"),
        ("matplotlib", "matplotlib"),
        ("openpyxl", "openpyxl"),
        ("umap", "umap-learn"),
        ("torch", "torch"),
    ]:
        importlib.import_module(module)
        print(distribution, version(distribution), flush=True)
    if args.dino_repo is not None:
        loader = importlib.import_module("2_embedding_extraction")
        repo, config, weights = loader.validate_dino_paths(
            args.dino_repo, args.dino_config, args.dino_weights
        )
        sys.path.insert(0, str(repo))
        import dinov2

        if not Path(dinov2.__file__).resolve().is_relative_to(repo):
            raise RuntimeError("A different dinov2 package shadows --dino-repo")
        for module in [
            "omegaconf",
            "fvcore",
            "iopath",
            "submitit",
            "torchvision",
            "torchmetrics",
            "monai",
            "nibabel",
            "torchio",
            "xformers.ops",
            "dinov2.eval.setup",
            "dinov2.configs",
        ]:
            importlib.import_module(module)
        from dinov2.configs import load_and_merge_config_3d

        load_and_merge_config_3d(str(config.with_suffix("")))
        print("3DINO imports and configuration passed:", repo, flush=True)
    import torch

    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable to this Python environment")
            torch.ones(1, device=device).sum().item()
            print(
                "CUDA:",
                torch.version.cuda,
                torch.cuda.get_device_name(device),
                flush=True,
            )
    if args.forward_check:
        if args.dino_repo is None:
            parser.error("--forward-check requires all three DINO path arguments")
        model = loader.load_dino(repo, config, weights, args.device or "cuda")
        with torch.inference_mode():
            result = model(
                torch.zeros((1, 1, 112, 112, 112), device=args.device or "cuda")
            )
        if (
            not isinstance(result, torch.Tensor)
            or result.shape != (1, 1024)
            or not torch.isfinite(result).all()
        ):
            raise RuntimeError("DINO did not return finite (1,1024) embeddings")
        print(
            "Synthetic DINO forward check passed; no biological data used.", flush=True
        )
    print("Environment checks passed.", flush=True)


if __name__ == "__main__":
    main()
