# Verification report

This is software verification using synthetic data, not biological validation.
No accuracy from these checks should be reported as a research result.

## Class metadata verification

Five focused tests passed: external mapping validation, numerical metric invariance
under renamed classes, QC/representation legend labels, launcher command order and
paths, and failure propagation. Preprocessing was repeated on 250 synthetic nuclei.
The saved mapping was checked through CV assembly, 40 reduced classical fits,
selection locks, validation/test exports, pooled statistics and representation
provenance. Existing synthetic feature arrays were reused for this metadata check;
their test-only provenance was associated with the newly generated synthetic
manifest. No production artifacts were altered. Both Figure 5 panels and PCA plots
were generated; a representation legend was visually inspected. UMAP was not rerun
for this change and uses the same tested legend function.

The complete PyTorch-dependent suite could not be repeated for this metadata change:
the available local PyTorch installation failed during import with a bus error.
CNN metadata propagation was inspected in code; CNN training and GPU inference
were not repeated. The earlier consolidated-environment checks below remain a
separate verification record. All updated Python files compiled and the launcher
passed Bash syntax validation.

## Consolidated-environment checks

- **17 tests passed** across the scientific and execution suites in the consolidated environment. These cover original-identity
  split isolation; exactly 10% validation in a divisible synthetic example;
  deterministic split generation; nested augmentation plans and per-animal/class
  counts; preservation of zero-intensity mask voxels; paired image/mask geometry;
  retained padding; 31-feature schema and finiteness; sphere radius/curvature;
  3D texture axis invariance and background exclusion; fixed-class metrics;
  training-only scaling and probabilities for all four classical models;
  explicit MLP validation; exact three-family paired statistical calculations;
  probability roundoff versus malformed probabilities; sequential runner order,
  paths containing spaces, failure propagation, DINO configuration path checks
  and numerical equality between serial and spawned feature workers.
- All seven numbered command-line entry points successfully ran `--help`.
- All Python files parsed/compiled successfully. A formatter was applied without
  changing numerical operations.
- A full reduced synthetic workflow completed: five source animals, five classes,
  10 objects per class per animal (250 original nuclei); preprocessing and split
  manifests; original and training-only handcrafted feature extraction; all five
  CV folds; original-only and two-row/class/animal augmentation conditions;
  40 classical candidate fits (four classifiers × two conditions × five folds),
  validation selection, saved-model reload and held-out evaluation.
- The same workflow generated original-only PCA/UMAP, both Figure 5 heatmaps,
  supplementary correlations, statistical CSV/Excel tables and SVG panels.
  Its 1024-dimensional vectors were random synthetic vectors, **not DINO outputs**.
- The original ResNet architecture completed one CPU fold with a one-epoch smoke
  budget, saved its validation checkpoint and reloaded it for validation/test
  evaluation. The final code repeated checkpoint evaluation successfully after
  probability-roundoff and patch-checksum safeguards were added.
- All 250 original image/mask pairs passed the final saved-file checksum checks.
- The Figure 5A SVG was rendered and inspected: all 31 rows and columns, variance
  labels, colorbar, white background and feature labels are present without clipping.

The single-family end-to-end statistics output and separate three-family paired
statistics test do not constitute a full real-data, three-family production run.

## Environment of the completed integration run

| Component | Version |
|---|---|
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| pandas | 2.2.3 |
| scikit-learn | 1.8.0 |
| scikit-image | 0.26.0 |
| tifffile | 2026.8.23 |
| PyTorch | 2.6.0+cu124 (CPU execution in this test) |
| UMAP | 0.5.12 |

The consolidated unit-test run had three nonfailure warnings: one scikit-learn warning
about future removal of the original LogisticRegression `penalty` parameter,
and two MLP batch-size clipping warnings caused by the deliberately tiny fixture.
The numerical dependency pin preserves the original grid interface. Do not
silently upgrade scikit-learn for an ongoing experiment.

The consolidated requirements were installed successfully in a fresh environment;
`pip check` reported no broken requirements. The complete synthetic workflow was
rerun after consolidating the scripts: preprocessing, feature extraction, five-fold
classical fitting/evaluation, representation plots, statistical tables and one
CNN fold all completed. No production-data performance is inferred from this test.
Dependency/import checks are described in `DEPENDENCIES.md`; full installed
versions are listed in `ENVIRONMENT.txt`.

## Not verified here

No original DAPI volumes, source masks, pretrained weights or target GPUs were
provided. Upstream 3DINO source and configuration imports were tested in the
consolidated environment (see `DEPENDENCIES.md`). Real DINO inference,
optional DINO-head fine-tuning, CUDA mixed precision, multiple GPUs, full 112³
training, all production hyperparameter fits and six production augmentation
sizes were not executed. Checkpoint and CUDA kernel compatibility must be
checked using the server checkout and weights.

The curvature descriptor is a documented finite-resolution numerical estimate.
A sphere sanity check is not proof of unbiased curvature on arbitrary nuclear
surfaces. Complexity descriptors are finite-scale proxies, not demonstrated
asymptotic fractal dimensions. Inspect feature distributions and mask QC before
committing the predictive protocol, using development data only.

The software blocks common identity, hash, schema and output-mixing errors. It
cannot enforce that a person never uses previously viewed test results to change
an experiment. Freeze the full protocol before final evaluation and report any
subsequent exploratory changes transparently.

## Reproduce

Use the installation and commands in README. Run unit tests first, then
`tests/synthetic_workflow.py --output NEW_EMPTY_PATH --cnn` under one numerical
thread. The script requires a new output directory and labels its results as
synthetic. Smoke-mode statistics require explicit `--allow-smoke`.

Keep the full real-run environment, source revision, manifests, feature plans,
selection locks, checkpoints and predictions together. `RELEASE_SHA256SUMS.txt`
records the delivered source/document files; it is not a validation of future
experimental outputs.

## Annotation update verification

A synthetic three-family comparison was plotted with three pairwise brackets.
The displayed Holm p-values match the paired-test table; SVG rendering was
inspected. These are plotting fixtures, not biological results.
