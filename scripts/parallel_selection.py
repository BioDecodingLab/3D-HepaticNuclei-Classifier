"""Resource-bounded classical selection with deterministic reduction and checkpoints.

One worker owns a complete (family, fold, level, estimator) search. Candidates
retain ParameterGrid order. Only the parent locks the final selection.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import joblib
from joblib import Memory, Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from classical_models import train_candidate
from evaluation import score_tuple
from model_grids import PARAM_GRIDS
from pipeline_common import (
    check_features, class_mapping, environment, read_json, require_new, sha256, write_json,
)


def _atomic_dump(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, tmp)
    os.replace(tmp, path)


def _condition(task, a, signature):
    dest = a.output / task["dest"]
    done = dest / "completed.json"
    if done.exists():
        record = read_json(done)
        if sha256(dest / "model.joblib") != record["model_sha256"]:
            raise ValueError(f"Completed model changed: {dest}")
        print(f"REUSE {task['dest']}", flush=True)
        return record
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = dest / "candidate_checkpoint.joblib"
    state = joblib.load(checkpoint) if checkpoint.exists() else dict(
        next_candidate=0, best=None, model=None, history=[], search=[]
    )
    file = a.cv / task["cv_file"]
    # Selection reads train and validation arrays only.
    with np.load(file, allow_pickle=False) as data:
        X, y, V, yv, ids, Xo, yo, ido = [
            data[k] for k in (
                "X_train", "y_train", "X_val", "y_val", "ids_train",
                "X_train_original", "y_train_original", "ids_train_original"
            )
        ]
        if set(ids) & set(data["ids_val"]):
            raise ValueError("Training-validation identity overlap")
    check_features(X)
    check_features(V)
    cache = (a.cache_dir or (a.output / ".preprocessing_cache")) / signature / task["dest"]
    memory = None if a.no_cache else Memory(cache, verbose=0)
    grid = list(ParameterGrid(PARAM_GRIDS[task["model"]]))
    if a.smoke:
        grid = grid[:1]
    with threadpool_limits(limits=a.threads_per_job):
        for i in range(state["next_candidate"], len(grid)):
            started = time.monotonic()
            p = grid[i]
            print(f"START {task['dest']} candidate {i + 1}/{len(grid)}", flush=True)
            model, m, history = train_candidate(
                task["model"], p, X, y, V, yv, ids, Xo, yo, ido, a.seed,
                max_epochs=2 if a.smoke else 500, patience=15, memory=memory
            )
            state["search"].append(dict(
                candidate=i, params=json.dumps(p, sort_keys=True), **m
            ))
            if state["best"] is None or score_tuple(m) > score_tuple(state["best"]["metrics"]):
                state.update(best=dict(metrics=m, params=p, candidate=i), model=model, history=history)
            state["next_candidate"] = i + 1
            # Best model and completed search prefix are committed together.
            _atomic_dump(state, checkpoint)
            del model
            print(
                f"DONE {task['dest']} candidate {i + 1}/{len(grid)} "
                f"in {time.monotonic() - started:.1f}s; validation macro-F1={m['f1_macro']:.6f}",
                flush=True
            )
    _atomic_dump(state["model"], dest / "model.joblib")
    pd.DataFrame(state["history"]).to_csv(dest / "training_history.csv", index=False)
    pd.DataFrame(state["search"]).to_csv(dest / "hyperparameter_search_results.csv", index=False)
    record = dict(
        family=task["family"], fold=task["fold"], model=task["model"], level=task["level"],
        **state["best"], artifact=str(Path(task["dest"]) / "model.joblib"),
        model_sha256=sha256(dest / "model.joblib"),
        cv_file=task["cv_file"], cv_sha256=task["cv_sha256"], selected=False
    )
    write_json(done, record)
    checkpoint.unlink(missing_ok=True)
    if not a.no_cache:
        shutil.rmtree(cache, ignore_errors=True)
    return record


def _select_locked(a):
    provenance = read_json(a.cv / "provenance.json")
    families = a.families or provenance["families"]
    tasks = []
    for family in families:
        for foldpath in sorted((a.cv / family).glob("test_*")):
            if a.folds and foldpath.name not in a.folds:
                continue
            reference = None
            for file in sorted(foldpath.glob("level_*.npz"), key=lambda p: int(p.stem.split("_")[-1])):
                level = int(file.stem.split("_")[-1])
                if a.levels is not None and level not in a.levels:
                    continue
                with np.load(file, allow_pickle=False) as data:
                    current = [data[k] for k in ("ids_val", "X_val", "y_val")]
                    if reference is not None and not all(np.array_equal(x, y) for x, y in zip(reference, current)):
                        raise ValueError("Validation changed between levels")
                    reference = current
                    if set(data["ids_train"]) & set(current[0]):
                        raise ValueError("Training-validation identity overlap")
                digest = sha256(file)
                for name in a.models:
                    tasks.append(dict(
                        family=family, fold=foldpath.name, model=name, level=level,
                        dest=str(Path(family) / foldpath.name / f"{name}_level_{level}"),
                        cv_file=str(file.relative_to(a.cv)), cv_sha256=digest,
                    ))
    if not tasks or len({t["dest"] for t in tasks}) != len(tasks):
        raise ValueError("Empty or duplicate condition selection")
    source = Path(__file__).parent
    config = dict(
        schema=1, tasks=tasks, seed=a.seed, smoke=a.smoke,
        threads_per_job=a.threads_per_job, environment=environment(),
        joblib_version=joblib.__version__,
        grids=json.loads(json.dumps(PARAM_GRIDS)),
        code={name: sha256(source / name) for name in (
            "4_run_models.py", "parallel_selection.py", "classical_models.py",
            "model_grids.py", "evaluation.py", "pipeline_common.py"
        )},
        cv_provenance_sha256=sha256(a.cv / "provenance.json"),
    )
    config = json.loads(json.dumps(config))
    config_path = a.output / "search_config.json"
    if config_path.exists():
        if not a.resume:
            raise FileExistsError("Existing search: use --resume or a new output directory")
        if read_json(config_path) != config:
            raise ValueError("Resume refused: inputs, grid, code, environment or numerical thread settings changed")
    else:
        require_new(a.output)
        write_json(config_path, config)
    if (a.output / "selection.json").exists():
        lock = read_json(a.output / "selection.json")
        if lock.get("search_config_sha256") != sha256(config_path):
            raise ValueError("Selection/config mismatch")
        for record in lock["selection"]:
            if sha256(a.output / record["artifact"]) != record["model_sha256"]:
                raise ValueError("Locked model changed")
        print("Selection already complete and verified.", flush=True)
        return
    signature = hashlib.sha256(json.dumps([str(a.output), config], sort_keys=True).encode()).hexdigest()[:20]
    # Launch expensive SVM levels first across all folds. Reduction below always
    # uses the original family/fold/level/model order, including exact ties.
    scheduled = sorted(tasks, key=lambda t: (t["model"] != "svm", -t["level"]))
    count = min(a.jobs, len(tasks))
    print(f"{len(tasks)} condition searches; {count} processes x {a.threads_per_job} threads", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=a.threads_per_job):
        results = Parallel(n_jobs=count, batch_size=1, pre_dispatch=count)(
            delayed(_condition)(task, a, signature) for task in scheduled
        )
    by_key = {(r["family"], r["fold"], r["level"], r["model"]): r for r in results}
    selection = [by_key[(t["family"], t["fold"], t["level"], t["model"])] for t in tasks]
    winners = {}
    for r in selection:
        key = r["family"], r["fold"]
        if key not in winners or score_tuple(r["metrics"]) > score_tuple(winners[key]["metrics"]):
            winners[key] = r
    for r in winners.values():
        r["selected"] = True
    write_json(a.output / "selection.json", dict(
        selection=selection, class_names=class_mapping(provenance),
        smoke_test=a.smoke, seed=a.seed, environment=environment(),
        cv_provenance_sha256=config["cv_provenance_sha256"],
        search_config_sha256=sha256(config_path),
        protocol="train-only fit; clean validation selection; no train+validation refit",
    ))
    print("Selection locked. Run --phase evaluate only after all model searches are complete.", flush=True)


def select(a):
    # Defaults retain compatibility with programmatic calls from existing tests.
    for key, default in dict(jobs=1, threads_per_job=1, resume=False, cache_dir=None, no_cache=False).items():
        if not hasattr(a, key):
            setattr(a, key, default)
    if a.jobs < 1 or a.threads_per_job < 1:
        raise ValueError("--jobs and --threads-per-job must be positive")
    cpus = joblib.cpu_count()
    if a.jobs * a.threads_per_job > cpus:
        raise ValueError(f"Requested CPU budget exceeds {cpus} CPUs available to this process")
    a.output = Path(a.output).resolve()
    a.cv = Path(a.cv).resolve()
    if a.cache_dir is not None:
        a.cache_dir = Path(a.cache_dir).resolve()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Persistent adjacent inode prevents simultaneous selectors, including direct CLI use.
    with (a.output.parent / (a.output.name + ".selection.lock")).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another selector is using this output directory") from exc
        _select_locked(a)
