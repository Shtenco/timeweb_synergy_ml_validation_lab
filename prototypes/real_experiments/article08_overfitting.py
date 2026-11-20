"""Measure a train/validation/test gap on UCI's Wisconsin Diagnostic dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def _hash_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def run_real_experiment(
    seeds: tuple[int, ...] = (7, 17, 27),
    train_fractions: tuple[float, ...] = (0.15, 0.4, 1.0),
    max_iter: int = 150,
) -> dict:
    if not seeds or max_iter < 1:
        raise ValueError("Provide at least one seed and max_iter >= 1")
    if any(not 0.05 <= fraction <= 1.0 for fraction in train_fractions):
        raise ValueError("train_fractions must be between 0.05 and 1.0")

    dataset = load_breast_cancer()
    X = np.asarray(dataset.data, dtype=np.float64)
    y = np.asarray(dataset.target, dtype=np.int64)
    indices = np.arange(len(y), dtype=np.int64)
    train_idx, remainder_idx = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=20261117
    )
    validation_idx, test_idx = train_test_split(
        remainder_idx, test_size=0.50, stratify=y[remainder_idx], random_state=20261118
    )
    if set(train_idx) & set(validation_idx) or set(train_idx) & set(test_idx) or set(validation_idx) & set(test_idx):
        raise AssertionError("train, validation, and test partitions must be disjoint")

    runs: list[dict] = []
    with threadpool_limits(limits=1):
        for fraction in train_fractions:
            if fraction == 1.0:
                subset_idx = train_idx
            else:
                subset_idx, _ = train_test_split(
                    train_idx,
                    train_size=fraction,
                    stratify=y[train_idx],
                    random_state=20261200 + round(fraction * 1000),
                )
            for seed in seeds:
                for alpha in (0.0001, 1.0):
                    model = make_pipeline(
                        StandardScaler(),
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            solver="adam",
                            alpha=alpha,
                            learning_rate_init=0.003,
                            max_iter=max_iter,
                            random_state=seed,
                            early_stopping=False,
                            tol=1e-5,
                        ),
                    )
                    start = time.perf_counter()
                    model.fit(X[subset_idx], y[subset_idx])
                    fit_seconds = time.perf_counter() - start
                    classifier = model[-1]

                    def metrics(partition: np.ndarray, label: str) -> dict:
                        truth = y[partition]
                        probabilities = model.predict_proba(X[partition])[:, 1]
                        predictions = (probabilities >= 0.5).astype(np.int64)
                        return {
                            f"{label}_rows": int(len(partition)),
                            f"{label}_accuracy": float(accuracy_score(truth, predictions)),
                            f"{label}_auc": float(roc_auc_score(truth, probabilities)),
                            f"{label}_log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
                        }

                    row = {
                        "train_fraction": float(fraction),
                        "seed": int(seed),
                        "alpha": float(alpha),
                        "fit_seconds": fit_seconds,
                        "iterations": int(classifier.n_iter_),
                        "converged_before_max_iter": classifier.n_iter_ < max_iter,
                        **metrics(subset_idx, "train"),
                        **metrics(validation_idx, "validation"),
                        **metrics(test_idx, "test"),
                    }
                    row["train_test_auc_gap"] = row["train_auc"] - row["test_auc"]
                    runs.append(row)

    try:
        processor = platform.processor() or platform.machine()
    except Exception:
        processor = platform.machine()
    return {
        "article_id": "08",
        "experiment": "Overfit a fixed public binary-classification dataset and measure generalization gap",
        "evidence_type": "measured_public_dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "source_id": "UCI-17-WDBC",
            "source_url": "https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic",
            "loader": "sklearn.datasets.load_breast_cancer",
            "rows": int(len(y)),
            "features": int(X.shape[1]),
            "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "feature_sha256": _hash_array(X),
            "target_sha256": _hash_array(y),
            "license": "CC BY 4.0 (UCI dataset record)",
        },
        "method": {
            "model": "sklearn MLPClassifier hidden_layer_sizes=(64, 32), StandardScaler fit on training subset only",
            "split": "stratified 70% train / 15% validation / 15% held-out test; indices fixed before model fitting",
            "train_fractions_of_train_partition": list(train_fractions),
            "seeds": list(seeds),
            "alpha_values": [0.0001, 1.0],
            "max_iter": max_iter,
            "selection_rule": "No hyperparameter or checkpoint selection on the held-out test set",
            "metrics": ["accuracy", "ROC AUC", "log loss", "train-test AUC gap", "fit wall time"],
        },
        "split_rows": {
            "train_partition": int(len(train_idx)),
            "validation": int(len(validation_idx)),
            "held_out_test": int(len(test_idx)),
            "overlap_count": 0,
            "train_partition_indices_sha256": _hash_array(np.sort(train_idx)),
            "validation_indices_sha256": _hash_array(np.sort(validation_idx)),
            "test_indices_sha256": _hash_array(np.sort(test_idx)),
        },
        "split_overlap_count": 0,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": processor,
            "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
            "thread_limit": 1,
        },
        "runs": runs,
        "limitations": [
            "This public, small tabular benchmark is not a clinical validation study and must not guide medical decisions.",
            "It measures overfitting/generalization on this dataset only; it does not test temporal leakage, duplicate leakage, or group leakage.",
            "A random patient-level split from this dataset cannot establish external or prospective generalization.",
            "Wall time is machine-specific; no performance claim should be generalized from this laptop.",
        ],
    }


def save_result(result: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"article-08-wdbc-overfit-{timestamp}"
    json_path = output_dir / f"{stem}.json"
    with json_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["runs"][0]))
        writer.writeheader()
        writer.writerows(result["runs"])
    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("evidence/08_real"))
    parser.add_argument("--max-iter", type=int, default=150)
    args = parser.parse_args(argv)
    result = run_real_experiment(max_iter=args.max_iter)
    path = save_result(result, args.out)
    print(path)
    for fraction in result["method"]["train_fractions_of_train_partition"]:
        rows = [row for row in result["runs"] if row["train_fraction"] == fraction]
        print(
            f"train_fraction={fraction:.2f} "
            f"test_auc={np.mean([row['test_auc'] for row in rows]):.4f} "
            f"train_auc={np.mean([row['train_auc'] for row in rows]):.4f} "
            f"fit_seconds_mean={np.mean([row['fit_seconds'] for row in rows]):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
