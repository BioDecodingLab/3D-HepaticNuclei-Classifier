# Dependency specification

Install the consolidated `requirements.txt` inside `nuclei_classification`.
It covers this pipeline and the external 3DINO interfaces it uses.

## Supported platform

Python 3.12; Linux x86_64; glibc 2.28 or newer; an NVIDIA driver supporting the
CUDA 12.4 runtime for GPU execution. The xFormers wheel is explicitly selected
from the CUDA 12.4 index. This file is not a portable Windows/macOS/CPU-only lock.

| Component | Pin | Purpose |
|---|---|---|
| PyTorch | 2.6.0+cu124 | Neural networks and CUDA operations |
| torchvision | 0.21.0+cu124 | External 3DINO utilities |
| xFormers | 0.0.29.post3, CUDA 12.4 | 3DINO attention and layer operations |
| OmegaConf | 2.3.0 | YAML configuration |
| TorchMetrics | 0.10.3 | External 3DINO metric API |
| setuptools | 80.9.0 | Supplies `pkg_resources`, required by this TorchMetrics version |
| MONAI | 1.5.1 | External image utilities, compatible with this NumPy/PyTorch stack |
| NiBabel | 5.3.2 | Medical image IO |
| TorchIO | 0.20.23 | External 3D image transform imports |
| NumPy / SciPy | 2.3.5 / 1.17.0 | Numerical and statistical operations |
| scikit-learn | 1.8.0 | Classifiers, scaling and PCA |
| scikit-image | 0.26.0 | Resizing, region measurements and surface meshes |

fvcore, iopath, submitit, einops, plotting, TIFF IO, UMAP and testing packages
are also pinned in the consolidated file. The matching xFormers release series
is documented in its [upstream changelog](https://github.com/facebookresearch/xformers/blob/main/CHANGELOG.md).
The classification pipeline uses its own image/mask transforms; installing MONAI
does not substitute a different preprocessing procedure.

cuML and the NVIDIA Python package index are not required. Neither this pipeline
nor the inspected 3DINO model-loading path imports cuML. PCA, UMAP and classical
models use the CPU libraries specified here. CUDA 11 packages and xFormers 0.0.18
do not belong in this pinned CUDA 12.4 environment.

Obtain 3DINO source and weights separately and pass the source checkout through
`--dino-repo`. Do not install another requirements file over this environment.
Third-party source code and weights are not redistributed in this repository.

## Compatibility verification

A clean Python 3.12 environment was installed from `requirements.txt`.
`python -m pip check` reported no broken requirements. Imports passed for
PyTorch, torchvision, xFormers operations, MONAI, NiBabel, TorchIO, TorchMetrics,
and the upstream 3DINO loading/configuration/data-transform/metric modules.

The external source revision checked was
`85bd4435c1b2ada41cd34cd15cad17c4d3c88d89` from
[AICONSlab/3DINO](https://github.com/AICONSlab/3DINO).
The `train/vit3d_highres` configuration loaded as `vit_large_3d`.

CUDA kernel execution and checkpoint loading require verification on the target
GPU, since no NVIDIA GPU or biological checkpoint was available in this test:

```bash
python scripts/check_environment.py --device cuda \
  --dino-repo "$DINO_REPO" --dino-config "$DINO_CONFIG" \
  --dino-weights "$DINO_WEIGHTS" --forward-check
```

The upstream model builder uses CUDA internally. CPU smoke tests cover the
handcrafted/classical/CNN paths, but not pretrained DINO inference.
TorchMetrics emits a `pkg_resources` deprecation warning under this environment;
its import succeeds with the setuptools pin above.

The primary dependencies are pinned; pip resolves transitive dependencies.
Save `python -m pip freeze` with each experiment. `docs/ENVIRONMENT.txt` records
the complete verification environment, including transitive packages.
