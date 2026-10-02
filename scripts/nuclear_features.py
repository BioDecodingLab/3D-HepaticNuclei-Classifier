"""31 mask-defined descriptors in standardized voxel coordinates.

No Otsu, inferred intensity masks, or population-fitted preprocessing.
See docs/FEATURES.md for equations, units and estimator limitations.
"""

import itertools
import numpy as np
from scipy import ndimage as ndi, stats
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.morphology import ball

FEATURE_NAMES = [
    "voxel_count",
    "surface_area_voxels",
    "surface_area_mesh",
    "elongation",
    "sphericity",
    "mean_radius",
    "radius_variance",
    "shape_index_mean",
    "surface_to_volume_ratio",
    "cvm",
    "curvature_mean",
    "curvature_std",
    "curvature_skewness",
    "curvature_kurtosis",
    "intensity_mean",
    "intensity_std",
    "intensity_skewness",
    "intensity_kurtosis",
    "fractal_box_counting",
    "fractal_minkowski_bouligand",
    "weighted_lacunarity_1_5",
    "mean_surface_intensity_gradient",
    "haralick_contrast",
    "haralick_dissimilarity",
    "haralick_homogeneity",
    "haralick_ASM",
    "haralick_energy",
    "haralick_correlation",
    "bbox_volume",
    "extent",
    "equivalent_diameter",
]


def moments(x, weights=None):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return [np.nan] * 4
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    mean = np.sum(w * x)
    delta = x - mean
    var = np.sum(w * delta**2)
    if var < 1e-16:
        return float(mean), 0.0, 0.0, 0.0
    return (
        float(mean),
        float(np.sqrt(var)),
        float(np.sum(w * delta**3) / var**1.5),
        float(np.sum(w * delta**4) / var**2 - 3),
    )


def surface_geometry(mask, sigma=1.0):
    """Closed voxel-mask mesh; signed-distance curvature with outward normal.

    Smoothing affects curvature estimation only. Mesh area and radial distances
    use the original binary-mask isosurface. Principal curvature samples use the
    tangent-projected Hessian of a smoothed signed-distance field.
    """
    padded = np.pad(mask, 5)
    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float32), 0.5, allow_degenerate=False
    )
    area = float(mesh_surface_area(verts, faces))
    tri = verts[faces]
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    weights = np.zeros(len(verts))
    for i in range(3):
        np.add.at(weights, faces[:, i], triangle_area / 3)
    centroid = np.argwhere(padded).mean(axis=0)
    radius = np.linalg.norm(verts - centroid, axis=1)
    rm, rs, _, _ = moments(radius, weights)
    signed = ndi.distance_transform_edt(~padded) - ndi.distance_transform_edt(padded)
    signed = ndi.gaussian_filter(signed, sigma)
    grad = np.gradient(signed)
    coords = verts.T
    gv = np.stack([ndi.map_coordinates(g, coords, order=1) for g in grad], axis=1)
    norm = np.linalg.norm(gv, axis=1)
    valid = norm > 1e-8
    normal = gv / np.maximum(norm[:, None], 1e-8)
    hessian = np.empty((len(verts), 3, 3))
    for i in range(3):
        for j in range(3):
            hessian[:, i, j] = ndi.map_coordinates(
                np.gradient(grad[i], axis=j), coords, order=1
            )
    hessian = (hessian + hessian.transpose(0, 2, 1)) / 2
    axis = np.eye(3)[np.argmin(abs(normal), axis=1)]
    u = np.cross(normal, axis)
    u /= np.maximum(np.linalg.norm(u, axis=1)[:, None], 1e-8)
    v = np.cross(normal, u)
    basis = np.stack([u, v], axis=2)
    shape = np.einsum("nki,nkl,nlj->nij", basis, hessian, basis) / np.maximum(
        norm[:, None, None], 1e-8
    )
    k = np.linalg.eigvalsh(shape)
    k2, k1 = k[:, 0], k[:, 1]
    mean_curv = (k1 + k2) / 2
    cm, cs, csk, ck = moments(mean_curv[valid], weights[valid])
    curved = valid & (np.hypot(k1, k2) > 1e-8)
    shape_index = (
        2 / np.pi * np.arctan2(k1[curved] + k2[curved], k1[curved] - k2[curved])
    )
    si = (
        float(np.average(shape_index, weights=weights[curved]))
        if curved.any()
        else np.nan
    )
    return dict(
        surface_area_mesh=area,
        mean_radius=rm,
        radius_variance=rs**2,
        shape_index_mean=si,
        curvature_mean=cm,
        curvature_std=cs,
        curvature_skewness=csk,
        curvature_kurtosis=ck,
        cvm=cs**2,
    )


def glcm_3d(image, mask, levels=32):
    """Symmetric GLCMs: 13 offsets, masked pairs, fixed [0,1] quantization.

    Feature values are computed separately per direction then averaged.
    """
    q = np.minimum((np.clip(image, 0, 1) * levels).astype(int), levels - 1)
    offsets = [
        o
        for o in itertools.product([-1, 0, 1], repeat=3)
        if o != (0, 0, 0) and next(v for v in o if v) != -1
    ]
    ii, jj = np.indices((levels, levels))
    results = []
    for offset in offsets:
        a, b = [], []
        for n, d in zip(image.shape, offset):
            a.append(slice(max(0, -d), min(n, n - d)))
            b.append(slice(max(0, d), min(n, n + d)))
        a, b = tuple(a), tuple(b)
        valid = mask[a] & mask[b]
        if not valid.any():
            continue
        c = (
            np.bincount(q[a][valid] * levels + q[b][valid], minlength=levels**2)
            .reshape(levels, levels)
            .astype(float)
        )
        c = c + c.T
        c /= c.sum()
        mx = (c * ii).sum()
        my = (c * jj).sum()
        sx = np.sqrt((c * (ii - mx) ** 2).sum())
        sy = np.sqrt((c * (jj - my) ** 2).sum())
        asm = (c * c).sum()
        results.append(
            [
                (c * (ii - jj) ** 2).sum(),
                (c * abs(ii - jj)).sum(),
                (c / (1 + (ii - jj) ** 2)).sum(),
                asm,
                np.sqrt(asm),
                (
                    (c * (ii - mx) * (jj - my)).sum() / (sx * sy)
                    if sx * sy > 1e-12
                    else 1.0
                ),
            ]
        )
    if not results:
        raise ValueError("Mask has no neighboring voxel pairs")
    return dict(zip(FEATURE_NAMES[22:28], np.mean(results, axis=0)))


def box_masses(mask, b):
    pads = [(0, (-n) % b) for n in mask.shape]
    padded = np.pad(mask.astype(np.uint8), pads)
    z, y, x = padded.shape
    return padded.reshape(z // b, b, y // b, b, x // b, b).sum(axis=(1, 3, 5)).ravel()


def complexity(mask):
    sizes = np.array([s for s in [1, 2, 4, 8, 16] if s <= min(mask.shape)])
    counts = np.array([np.count_nonzero(box_masses(mask, int(s))) for s in sizes])
    boxfd = (
        float(np.polyfit(np.log(1 / sizes), np.log(counts), 1)[0])
        if len(sizes) > 1
        else np.nan
    )
    # Finite-scale exterior parallel-volume exponent; not an asymptotic fractal proof.
    pad = np.pad(mask, 5)
    volume = float(pad.sum())
    radii = np.arange(1, 5)
    shells = np.array(
        [ndi.binary_dilation(pad, structure=ball(int(r))).sum() - volume for r in radii]
    )
    mb = float(3 - np.polyfit(np.log(radii), np.log(shells), 1)[0])
    lac = []
    for b in range(1, 6):
        mass = box_masses(mask, b).astype(float)
        lac.append(np.mean(mass**2) / mass.mean() ** 2)
    return dict(
        fractal_box_counting=boxfd,
        fractal_minkowski_bouligand=mb,
        weighted_lacunarity_1_5=float(np.average(lac, weights=np.arange(1, 6))),
    )


def extract_features(image, mask):
    image = np.asarray(image, np.float32)
    mask = np.asarray(mask, bool)
    if image.shape != mask.shape or image.ndim != 3 or mask.sum() < 8:
        raise ValueError("Invalid nuclear mask")
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite image")
    coords = np.argwhere(mask)
    volume = float(len(coords))
    surface = mask & ~ndi.binary_erosion(mask)
    cov = np.cov(coords.T)
    eig = np.linalg.eigvalsh(cov)
    elongation = float(np.sqrt(max(eig[-1], 1e-8) / max(eig[0], 1e-8)))
    geom = surface_geometry(mask)
    intensity = moments(image[mask])
    bbox = float(np.prod(coords.max(axis=0) - coords.min(axis=0) + 1))
    grad = np.sqrt(sum(g * g for g in np.gradient(image)))
    values = dict(
        voxel_count=volume,
        surface_area_voxels=float(surface.sum()),
        elongation=elongation,
        sphericity=float(
            np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / geom["surface_area_mesh"]
        ),
        surface_to_volume_ratio=geom["surface_area_mesh"] / volume,
        intensity_mean=intensity[0],
        intensity_std=intensity[1],
        intensity_skewness=intensity[2],
        intensity_kurtosis=intensity[3],
        mean_surface_intensity_gradient=float(grad[surface].mean()),
        bbox_volume=bbox,
        extent=volume / bbox,
        equivalent_diameter=float((6 * volume / np.pi) ** (1 / 3)),
        **geom,
        **complexity(mask),
        **glcm_3d(image, mask)
    )
    result = np.array([values[n] for n in FEATURE_NAMES], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite descriptor; inspect geometry")
    return result
