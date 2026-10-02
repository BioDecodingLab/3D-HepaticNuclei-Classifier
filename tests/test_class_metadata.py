"""External display names must not change numerical evaluation."""
import json
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline_common import class_mapping
from evaluation import export_evaluation, metrics

ROOT = Path(__file__).resolve().parents[1]


def test_external_mapping_and_invalid_metadata():
    mapping = json.loads((ROOT / "config/class_names.json").read_text())
    assert list(class_mapping({"class_names": mapping}).values()) == [
        "Hepatocyte", "Stellate cell", "Kupffer cell", "Endothelial cell", "Other cell"
    ]
    assert class_mapping()["1"] == "Class 1"
    for bad in [{"1": "One"}, {str(i): "Repeated" for i in range(1, 6)},
                {**mapping, "2": " "}, list(mapping.values())]:
        with pytest.raises(ValueError):
            class_mapping({"class_names": bad})


def test_named_exports_preserve_metrics(tmp_path):
    mapping = {str(i): f"External type {i}" for i in range(1, 6)}
    y = np.tile(np.arange(1, 6), 2)
    prob = np.full((10, 5), 0.025)
    prob[np.arange(10), y - 1] = 0.9
    result = export_evaluation(tmp_path, y, y, prob, np.arange(10),
                               np.repeat("animal", 10), class_names=mapping)
    assert result == metrics(y, y, prob)
    frame = pd.read_csv(tmp_path / "predictions.csv")
    assert frame.true_label.tolist() == y.tolist()
    assert frame.true_class.tolist() == [mapping[str(k)] for k in y]
    assert "External type 2" in (tmp_path / "classification_report.txt").read_text()
    for file in tmp_path.glob("*.svg"):
        assert "External type 2" in file.read_text()


def test_inspection_and_representation_legends(tmp_path):
    mapping = json.loads((ROOT / "config/class_names.json").read_text())
    labels = np.tile(np.arange(1, 6), 2)
    table = pd.DataFrame({"animal": np.repeat(["a", "b"], 5), "label": labels})
    for col in ["depth", "height", "width", "bbox_voxels", "mask_voxels", "original_volume_um3"]:
        table[col] = np.arange(10) + 1
    importlib.import_module("1_preprocessing").plot_inspection(table, tmp_path / "qc", mapping)
    scatter = importlib.import_module("7_representation_analysis").scatter
    scatter(np.arange(20).reshape(10, 2), labels, table.animal.to_numpy(),
            tmp_path / "scatter.svg", "PC1", "PC2", "Synthetic", mapping)
    for path in [tmp_path / "qc/depth_by_class.svg", tmp_path / "qc/animal_class_counts.svg", tmp_path / "scatter.svg"]:
        for name in mapping.values():
            assert name in path.read_text()
