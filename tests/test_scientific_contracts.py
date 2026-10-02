import importlib
import numpy as np
import pandas as pd
import pytest
from dataset_helper import standardize_pair, augment_pair, center_pad_3d
from nuclear_features import extract_features, FEATURE_NAMES, glcm_3d
from pipeline_common import validate_split
from classical_models import train_candidate
from model_grids import PARAM_GRIDS
from sklearn.model_selection import ParameterGrid
from evaluation import metrics


def sphere(r=6, n=24):
    z, y, x = np.indices((n, n, n))
    return (z - n // 2) ** 2 + (y - n // 2) ** 2 + (x - n // 2) ** 2 <= r * r


def table():
    return pd.DataFrame(
        [
            dict(
                sample_id=f"a{a}_c{c}_{i}",
                animal=f"img_{a}",
                label=c,
                image_path="a",
                mask_path="m",
            )
            for a in range(5)
            for c in range(1, 6)
            for i in range(10)
        ]
    )


def test_shared_split_isolation_and_ten_percent():
    pre = importlib.import_module("1_preprocessing")
    t = table()
    folds = pre.create_splits(t)
    for f in folds:
        validate_split(t, f)
        assert len(f["validation"]) == 20
        assert len(f["train"]) == 180
        assert len(f["test"]) == 50
    assert folds == pre.create_splits(t)
    f = dict(folds[0])
    f["train"] = f["train"] + [f["test"][0]]
    with pytest.raises(ValueError):
        validate_split(t, f)


def test_augmented_plan_targets_and_nested_levels():
    t = table()
    fold = importlib.import_module("1_preprocessing").create_splits(t)[0]
    make = importlib.import_module("2_embedding_extraction").make_plan
    small, big = make(t, fold, 100, 42), make(t, fold, 200, 42)
    assert small == [r for r in big if r["draw"] < 100]
    assert set(r["sample_id"] for r in big) <= set(fold["train"])
    counts = pd.DataFrame(big).groupby(["animal", "label"]).size()
    assert (counts == 200).all()
    assert len({r["row_id"] for r in big}) == len(big)


def test_original_mask_keeps_zero_intensity_foreground():
    m = sphere()
    im = m.astype(np.float32)
    im[12, 12, 12] = 0
    v, mask = standardize_pair(im, m, initial=(24,) * 3, target=(24,) * 3)
    assert mask[12, 12, 12] and v[12, 12, 12] == 0
    assert np.array_equal(mask, m)
    v2, m2 = augment_pair(v, mask, 23)
    assert m2.sum() == mask.sum()
    assert np.all(v2[~m2] == 0)
    v3, m3 = augment_pair(v, mask, 23)
    assert np.array_equal(v2, v3) and np.array_equal(m2, m3)


def test_padding_contract():
    assert center_pad_3d(np.ones((8, 9, 10)), (70,) * 3).shape == (70,) * 3
    assert center_pad_3d(np.ones((71, 9, 10)), (70,) * 3).shape == (71,) * 3


def test_sphere_geometry_and_feature_schema():
    m = sphere()
    v = extract_features(m.astype(np.float32) * 0.7, m)
    f = dict(zip(FEATURE_NAMES, v))
    assert len(v) == 31 and np.isfinite(v).all()
    assert f["elongation"] == pytest.approx(1, abs=0.01)
    assert f["mean_radius"] == pytest.approx(6, rel=0.05)
    assert f["curvature_mean"] == pytest.approx(1 / 6, rel=0.15)
    assert f["shape_index_mean"] > 0.85
    assert f["voxel_count"] == m.sum()


def test_texture_mask_and_3d_axis_invariance():
    m = sphere()
    z, y, x = np.indices(m.shape)
    im = (z % 2).astype(np.float32)
    f = glcm_3d(im, m)
    g = glcm_3d(im.transpose(2, 1, 0), m.transpose(2, 1, 0))
    assert f == pytest.approx(g)
    assert f["haralick_contrast"] > 0
    changed = im.copy()
    changed[~m] = np.random.default_rng(1).random((~m).sum())
    assert f == pytest.approx(glcm_3d(changed, m))


def test_fixed_five_class_metrics():
    m = metrics(np.array([1, 1]), np.array([1, 1]))
    assert m["f1_macro"] == pytest.approx(0.2)
    assert m["balanced_accuracy"] == 1
    assert m["n_present_classes"] == 1


@pytest.mark.parametrize("name", ["logreg", "rf", "svm", "mlp"])
def test_train_only_scaling_and_probabilities(name):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(50, 8))
    y = np.repeat(np.arange(1, 6), 10)
    V = rng.normal(size=(10, 8)) + 100
    yv = np.repeat(np.arange(1, 6), 2)
    ids = np.array([f"s{i}" for i in range(50)])
    p = list(ParameterGrid(PARAM_GRIDS[name]))[0]
    model, m, h = train_candidate(name, p, X, y, V, yv, ids, X, y, ids, max_epochs=2)
    pipeline = model.pipeline if name == "svm" else model
    assert np.allclose(
        pipeline.named_steps["pre"].named_steps["scaler"].mean_, X.mean(0)
    )
    prob = model.predict_proba(V)
    assert prob.shape == (10, 5) and np.allclose(prob.sum(1), 1)
    assert len(h) == 2 if name == "mlp" else h == []


def test_three_family_paired_statistics():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    stats = importlib.import_module("6_statistical_analysis")
    frame = pd.DataFrame(
        [
            dict(fold=f, experiment=name, f1_macro=0.5 + 0.05 * i + 0.01 * f)
            for f in range(5)
            for i, name in enumerate(["DINO", "Handcrafted", "CNN"])
        ]
    )
    result = stats.paired_tests(frame, ["f1_macro"])
    assert len(result) == 3
    assert np.allclose(result.p_raw, 0.0625)
    assert np.all(result.p_holm >= result.p_raw)
    permutation = stats.friedman_permutation(frame)
    assert permutation["permutations"] == 7776
    assert 0 < permutation["p_value"] <= 1
    with pytest.raises(ValueError):
        stats.paired_tests(frame.iloc[1:], ["f1_macro"])


def test_probability_roundoff_and_invalid_probabilities():
    from evaluation import normalized_probabilities

    p = np.tile(np.array([0.1, 0.2, 0.3, 0.1, 0.3], dtype=np.float32), (5, 1))
    q = normalized_probabilities(p)
    assert np.allclose(q.sum(1), 1.0, atol=1e-14)
    with pytest.raises(ValueError):
        normalized_probabilities(p * 0.5)
