"""Execution-contract tests independent of GPU hardware or biological datasets."""

import importlib
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_runner_order_paths_and_no_dry_run_writes(tmp_path):
    results = tmp_path / "results with spaces"
    command = [
        "bash",
        str(ROOT / "run_pipeline.sh"),
        "--dry-run",
        "--data",
        str(tmp_path / "data with spaces"),
        "--results",
        str(results),
        "--data-parallel",
    ]
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    calls = [
        shlex.split(line)
        for line in run.stdout.splitlines()
        if line.startswith("python ")
    ]
    assert len(calls) == 9
    assert [Path(c[1]).name[:1] for c in calls] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "4",
        "5",
        "6",
        "7",
    ]
    assert calls[1][calls[1].index("--output") + 1] == str(results / "features")
    assert calls[1][calls[1].index("--dino-weights") + 1] == str(
        tmp_path / "data with spaces/3dino_vit_weights.pth"
    )
    assert "--data-parallel" in calls[4]
    assert not results.exists()


def test_runner_stops_at_first_failed_stage(tmp_path):
    fake = tmp_path / "fake_python"
    trace = tmp_path / "trace.txt"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$RUNNER_TEST_TRACE"\n'
        'case "$*" in *1_preprocessing.py*) exit 7;; esac\n'
    )
    fake.chmod(0o755)
    run = subprocess.run(
        [
            "bash",
            str(ROOT / "run_pipeline.sh"),
            "--python",
            str(fake),
            "--results",
            str(tmp_path / "results"),
            "--stop-after",
            "3",
        ],
        env={**os.environ, "RUNNER_TEST_TRACE": str(trace)},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 7
    assert "1_preprocessing.py" in trace.read_text()
    assert "2_embedding_extraction.py" not in trace.read_text()
    assert "Stage 1 failed" in run.stderr


def test_dino_config_stem_file_and_missing_weights(tmp_path):
    loader = importlib.import_module("2_embedding_extraction")
    repo = tmp_path / "3DINO"
    (repo / "dinov2/configs/train").mkdir(parents=True)
    config = repo / "dinov2/configs/train/example.yaml"
    config.write_text("student: {}\n")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"Path validation fixture, never loaded as weights")
    a = loader.validate_dino_paths(repo, config, weights)
    assert a == loader.validate_dino_paths(repo, config.with_suffix(""), weights)
    assert a == loader.validate_dino_paths(repo, Path("train/example"), weights)
    with pytest.raises(FileNotFoundError, match="weights"):
        loader.validate_dino_paths(repo, config, tmp_path / "missing.pth")


def test_spawned_feature_workers_match_serial(tmp_path):
    from argparse import Namespace
    import numpy as np
    import tifffile

    z, y, x = np.indices((24, 24, 24))
    mask = (z - 12) ** 2 + (y - 12) ** 2 + (x - 12) ** 2 <= 36
    tifffile.imwrite(tmp_path / "image.tif", mask.astype(np.float32) * 0.7)
    tifffile.imwrite(tmp_path / "mask.tif", mask.astype(np.uint8))
    row = dict(sample_id="synthetic", image_path="image.tif", mask_path="mask.tif")
    args = Namespace(initial=(24,) * 3, target=(24,) * 3, workers=0)
    extractor = importlib.import_module("2_embedding_extraction")
    serial = extractor.extract(tmp_path, [row], args, "handcrafted")
    args.workers = 2
    spawned = extractor.extract(tmp_path, [row], args, "handcrafted")
    np.testing.assert_array_equal(serial, spawned)
