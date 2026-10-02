# Handcrafted nuclear feature definitions

All classifier features use the standardized [0,1] DAPI intensity patch and its
transformed **original instance mask**. There is no intensity thresholding or
replacement mask. A standardized voxel is the unit of length below, not 0.3 µm.
Original physical nuclear volume is recorded separately by preprocessing.
Photometric augmentation changes intensity descriptors, but never the mask.
Axis permutations/reflections transform image and mask together.

Let M be the binary mask, V its voxel count, A its triangulated surface area,
and c its volume centroid. A closed isosurface is obtained from a padded binary
mask at level 0.5 using marching cubes. Vertices are weighted by one third of
incident triangle area. Surface distances and area use this unsmoothed mask mesh.

For curvature only, the signed Euclidean distance field (positive outside) is
Gaussian-smoothed with sigma=1 standardized voxel. Its gradient and Hessian are
interpolated at mesh vertices. The tangent-plane projection of Hessian/gradient
norm gives two principal-curvature estimates k1 >= k2. With outward normals,
convex spheres have positive curvature. Mesh resolution and smoothing influence
these estimates; they are numerical estimators, not exact continuous geometry.

| # | Column | Definition / unit |
|---|---|---|
| 1 | voxel_count | V; standardized voxel count |
| 2 | surface_area_voxels | Count of mask voxels removed by default six-connected binary erosion; boundary-voxel proxy, not area |
| 3 | surface_area_mesh | A; standardized voxel² |
| 4 | elongation | sqrt(largest/smallest covariance eigenvalue of foreground coordinates); eigenvalues floored at 1e-8 |
| 5 | sphericity | pi^(1/3) (6V)^(2/3) / A |
| 6 | mean_radius | Surface-area-weighted mean of distance from c to mesh vertices; standardized voxels |
| 7 | radius_variance | Surface-area-weighted variance of those distances; standardized voxel² |
| 8 | shape_index_mean | Area-weighted mean of (2/pi) atan2(k1+k2, k1-k2); dimensionless, convex sphere approaches +1 |
| 9 | surface_to_volume_ratio | A/V; inverse standardized voxels |
| 10 | cvm | Weighted variance of mean curvature; inverse standardized voxel² |
| 11 | curvature_mean | Area-weighted mean H=(k1+k2)/2; inverse standardized voxels |
| 12 | curvature_std | Area-weighted standard deviation of H |
| 13 | curvature_skewness | Weighted third central moment / variance^(3/2) |
| 14 | curvature_kurtosis | Weighted fourth central moment / variance² minus 3 |
| 15 | intensity_mean | Mean [0,1] DAPI intensity over all mask voxels, including zero-valued foreground |
| 16 | intensity_std | Population standard deviation over mask voxels |
| 17 | intensity_skewness | Population standardized third central moment |
| 18 | intensity_kurtosis | Population standardized fourth central moment minus 3 |
| 19 | fractal_box_counting | Finite-scale slope of log occupied boxes against log(1/box size), sizes 1,2,4,8,16 |
| 20 | fractal_minkowski_bouligand | Finite-scale boundary parallel-volume descriptor: 3 minus slope of log exterior dilation-shell volume against log radius, radii 1,2,3,4 |
| 21 | weighted_lacunarity_1_5 | Average E[mass²]/E[mass]² for nonoverlapping boxes of sizes 1..5, weighted by box size |
| 22 | mean_surface_intensity_gradient | Mean central-difference gradient magnitude on boundary voxels |
| 23 | haralick_contrast | Mean directional sum p(i,j)(i-j)² |
| 24 | haralick_dissimilarity | Mean directional sum p(i,j) abs(i-j) |
| 25 | haralick_homogeneity | Mean directional sum p(i,j)/(1+(i-j)²) |
| 26 | haralick_ASM | Mean directional sum p(i,j)² |
| 27 | haralick_energy | Mean directional sqrt(ASM) |
| 28 | haralick_correlation | Mean directional covariance of gray levels divided by marginal standard deviations |
| 29 | bbox_volume | Product of mask bounding-box extents; standardized voxel³ |
| 30 | extent | V / bbox_volume |
| 31 | equivalent_diameter | (6V/pi)^(1/3); standardized voxels |

## Texture details

Quantization is fixed, q=min(floor(32 I),31), on normalized I in [0,1]. GLCMs use
all nearest neighbors in 3D, with 13 unique directions and their reverses via
symmetrization. A pair is counted only when both voxels belong to M. Directional
matrices are normalized separately, descriptor values calculated per direction,
and valid directions averaged equally. Directions without pairs are skipped;
absence of any valid pair is an error. Correlation of a constant directional
matrix is defined as 1, following the usual constant-texture convention.

The implementation does not require PyRadiomics at runtime. The direction convention agrees with
[the 3D GLCM description](https://pyradiomics.readthedocs.io/en/v3.0.1/_modules/radiomics/glcm.html).

## Geometry and complexity interpretation

A conventional shape index depends on the two principal curvatures. Near-flat locations (curvature norm below
1e-8) have undefined shape index and are excluded from its mean; no valid locations
is an extraction failure. The underlying curvature sign convention is fixed.
Smoothing scale is fixed in advance, never tuned using held-out performance.

Radius is a centroid-to-surface descriptor, not an equivalent-sphere radius or
a ray-intersection model. Non-star-convex nuclei remain measurable, but the
interpretation differs from local thickness.

The two columns carrying `fractal` in their names are **finite-scale
complexity descriptors**. A short log-log fit on a voxel object does not establish
biological fractality. In particular, the boundary parallel-volume descriptor
uses exterior shells with ample padding to avoid image-boundary truncation; its
finite-scale value need not equal the theoretical dimension 2 of a smooth surface.
Do not present it as a rigorously established Minkowski–Bouligand dimension.
Box counting and lacunarity depend on box origin, padding and scale range. These
are operational descriptors. Curvature variance `cvm` is
also an operational name; it is not a new named biological quantity.

All moments here use population-moment definitions; constant distributions have
standard deviation, skewness and excess kurtosis set to 0 as a numerical convention.
No population-dependent imputation, feature removal or mask repair occurs. Invalid
inputs and nonfinite outputs fail with the nucleus ID for investigation.

## Measurement and interpretation conventions

- Isotropic original spacing 0.3 µm; exact instance-mask provenance and label rules.
- Normalization and pad/resize operations, including variable scale for
  nuclei exceeding the initial cube. Standardized features are not original µm.
- Six resampled training targets per class per animal, with replacement and fixed
  shared original validation identities; no train+validation refit.
- Mask-defined morphology, full 3D masked texture and curvature estimators.
- Frozen DINO representation vs optional fine-tuned DINO branch distinction.
- Fold-specific validation selection; independent test evaluation after locking.
- Exploratory original-only PCA/UMAP and correlations; class/animal composition
  limits interpretation; no significance based on nuclei as independent animals.
