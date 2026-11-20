"""Compare row-wise and subject-wise evaluation on the public UCI HAR data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def extract_uci_har_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract the official nested UCI archive without writing outside output_dir."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    data_dir = output_dir / "UCI HAR Dataset"
    required = (
        data_dir / "train" / "X_train.txt",
        data_dir / "train" / "y_train.txt",
        data_dir / "train" / "subject_train.txt",
        data_dir / "test" / "X_test.txt",
        data_dir / "test" / "y_test.txt",
        data_dir / "test" / "subject_test.txt",
    )
    if all(path.is_file() for path in required):
        return data_dir
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix with a partial dataset directory: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as outer:
        try:
            nested_archive = outer.open("UCI HAR Dataset.zip")
        except KeyError as exc:
            raise ValueError("Expected nested member 'UCI HAR Dataset.zip'") from exc
        with nested_archive, ZipFile(nested_archive) as inner:
            for info in inner.infolist():
                if info.is_dir():
                    continue
                name = PurePosixPath(info.filename)
                if name.parts[0] == "__MACOSX":
                    continue
                if not name.parts or name.parts[0] != "UCI HAR Dataset":
                    raise ValueError(f"Unexpected path in UCI archive: {info.filename}")
                if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                    raise ValueError(f"Unsafe path in UCI archive: {info.filename}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Symlinks are not allowed in UCI archive: {info.filename}")
                relative = Path(*name.parts[1:])
                target = data_dir / relative
                if not target.resolve().is_relative_to(data_dir.resolve()):
                    raise ValueError(f"Unsafe path in UCI archive: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with inner.open(info) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
    if not all(path.is_file() for path in required):
        raise ValueError("Extracted archive is missing one or more official train/test files")
    return data_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_splits(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    official_train_subjects: set[int],
    official_test_subjects: set[int],
    seeds: tuple[int, ...] = (7, 17, 27),
    tree_count: int = 100,
) -> dict:
    X = np.asarray(X)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(subjects):
        raise ValueError("X, y, and subjects must have matching rows")
    if not seeds or tree_count < 1:
        raise ValueError("Provide at least one seed and tree_count >= 1")
    if official_train_subjects & official_test_subjects:
        raise ValueError("Official subject partitions must be disjoint")
    train_mask = np.isin(subjects, list(official_train_subjects))
    test_mask = np.isin(subjects, list(official_test_subjects))
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise ValueError("Official subject sets must cover every row exactly once")

    runs: list[dict] = []
    total_rows = np.arange(len(y))
    for seed in seeds:
        random_train, random_test = train_test_split(
            total_rows,
            test_size=0.30,
            random_state=seed,
            stratify=y,
        )
        partitions = (
            ("random_row_split", random_train, random_test),
            ("official_subject_split", np.flatnonzero(train_mask), np.flatnonzero(test_mask)),
        )
        for split_name, train_idx, test_idx in partitions:
            model = RandomForestClassifier(
                n_estimators=tree_count,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=seed,
            )
            started = time.perf_counter()
            model.fit(X[train_idx], y[train_idx])
            fit_seconds = time.perf_counter() - started
            predictions = model.predict(X[test_idx])
            train_subject_ids = sorted(int(x) for x in np.unique(subjects[train_idx]))
            test_subject_ids = sorted(int(x) for x in np.unique(subjects[test_idx]))
            overlap = sorted(set(train_subject_ids) & set(test_subject_ids))
            runs.append(
                {
                    "split": split_name,
                    "seed": int(seed),
                    "trees": int(tree_count),
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "train_subjects": train_subject_ids,
                    "test_subjects": test_subject_ids,
                    "subject_overlap_count": len(overlap),
                    "overlapping_subjects": overlap,
                    "accuracy": float(accuracy_score(y[test_idx], predictions)),
                    "macro_f1": float(f1_score(y[test_idx], predictions, average="macro")),
                    "fit_seconds": fit_seconds,
                }
            )

    return {
        "article_id": "08",
        "experiment": "Row-wise versus held-out-subject evaluation for activity recognition",
        "evidence_type": "measured_public_dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "model": f"RandomForestClassifier(n_estimators={tree_count}, max_features=sqrt, n_jobs=1)",
            "random_split": "stratified 70/30 row-wise split over all observations",
            "subject_split": "official UCI split, train/test subjects kept disjoint",
            "seeds": list(seeds),
            "no_tuning_on_test": True,
            "metrics": ["accuracy", "macro-F1", "train/test subject overlap", "fit wall time"],
        },
        "runs": runs,
        "limitations": [
            "The row-wise split deliberately violates the official subject-independent evaluation design.",
            "Windows are overlapping time-series segments, so random splitting may also place near-adjacent windows on both sides.",
            "This compares one feature-based Random Forest configuration on one dataset; it does not estimate performance in a different population or device setup.",
        ],
    }


def load_uci_har(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int], set[int], dict]:
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    required = (
        train_dir / "X_train.txt",
        train_dir / "y_train.txt",
        train_dir / "subject_train.txt",
        test_dir / "X_test.txt",
        test_dir / "y_test.txt",
        test_dir / "subject_test.txt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("UCI HAR files not found: " + ", ".join(missing))

    X_train = np.loadtxt(required[0], dtype=np.float32)
    y_train = np.loadtxt(required[1], dtype=np.int64)
    subjects_train = np.loadtxt(required[2], dtype=np.int64)
    X_test = np.loadtxt(required[3], dtype=np.float32)
    y_test = np.loadtxt(required[4], dtype=np.int64)
    subjects_test = np.loadtxt(required[5], dtype=np.int64)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    subjects = np.concatenate((subjects_train, subjects_test))
    provenance = {
        "source_id": "UCI-240-UCI-HAR",
        "source_url": "https://archive.ics.uci.edu/dataset/240/human-activity-recognition-using-smartphones",
        "citation": "Reyes-Ortiz et al. (2013), UCI Machine Learning Repository, DOI 10.24432/C54S4K",
        "license": "CC BY 4.0",
        "rows": int(len(y)),
        "features": int(X.shape[1]),
        "classes": sorted(int(x) for x in np.unique(y)),
        "subjects": sorted(int(x) for x in np.unique(subjects)),
        "official_train_rows": int(len(y_train)),
        "official_test_rows": int(len(y_test)),
        "input_sha256": {str(path.relative_to(data_dir)): _sha256(path) for path in required},
    }
    return (
        X,
        y,
        subjects,
        set(int(x) for x in np.unique(subjects_train)),
        set(int(x) for x in np.unique(subjects_test)),
        provenance,
    )


def run_real_experiment(data_dir: Path, seeds: tuple[int, ...] = (7, 17, 27), tree_count: int = 100) -> dict:
    X, y, subjects, official_train, official_test, provenance = load_uci_har(data_dir)
    result = compare_splits(X, y, subjects, official_train, official_test, seeds, tree_count)
    try:
        processor = platform.processor() or platform.machine()
    except Exception:
        processor = platform.machine()
    result["dataset"] = provenance
    result["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": processor,
        "logical_cpus": os.cpu_count(),
        "physical_memory_bytes": psutil.virtual_memory().total,
        "numpy": version("numpy"),
        "scikit_learn": version("scikit-learn"),
        "thread_limit": 1,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("evidence/08_real/data/UCI HAR Dataset"))
    parser.add_argument("--archive", type=Path, help="outer nested UCI ZIP; extract it before running")
    parser.add_argument("--out", type=Path, default=Path("evidence/08_real"))
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args(argv)
    if args.archive:
        args.data_dir = extract_uci_har_archive(args.archive, args.data_dir.parent)
    result = run_real_experiment(args.data_dir, tree_count=args.trees)
    args.out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = args.out / f"article-08-uci-har-leakage-{timestamp}.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(path)
    for split in ("random_row_split", "official_subject_split"):
        rows = [row for row in result["runs"] if row["split"] == split]
        print(
            f"{split}: accuracy={np.mean([x['accuracy'] for x in rows]):.4f} "
            f"macro_f1={np.mean([x['macro_f1'] for x in rows]):.4f} "
            f"subject_overlap={max(x['subject_overlap_count'] for x in rows)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
