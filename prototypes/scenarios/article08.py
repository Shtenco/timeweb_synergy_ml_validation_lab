"""Demonstrate entity leakage from row-wise splits with an exact memorizer."""

from __future__ import annotations

import random
from collections import Counter


def run(seed: int = 42, groups: int = 30, rows_per_group: int = 4) -> dict:
    if groups < 4 or rows_per_group < 2:
        raise ValueError("groups must be at least 4 and rows_per_group at least 2")
    rng = random.Random(seed)
    labels = {group: group % 2 for group in range(groups)}
    row_train = set()
    row_test = set()
    for group in range(groups):
        rows = list(range(rows_per_group))
        rng.shuffle(rows)
        row_test.add((group, rows[0]))
        row_train.update((group, row) for row in rows[1:])
    random_overlap = {g for g, _ in row_train} & {g for g, _ in row_test}
    row_lookup = {g: labels[g] for g, _ in row_train}
    random_accuracy = sum(row_lookup[g] == labels[g] for g, _ in row_test) / len(row_test)
    split = max(1, int(groups * 0.8))
    grouped_train = set(range(split))
    grouped_test = set(range(split, groups))
    group_overlap = grouped_train & grouped_test
    majority = Counter(labels[g] for g in grouped_train).most_common(1)[0][0]
    grouped_accuracy = sum(labels[g] == majority for g in grouped_test) / len(grouped_test)
    return {
        "evidence_type": "synthetic",
        "seed": seed,
        "random_row_split": {
            "accuracy": random_accuracy,
            "group_overlap_count": len(random_overlap),
            "test_rows": len(row_test),
            "predictor": "exact group-ID memorizer",
        },
        "group_split": {
            "accuracy": grouped_accuracy,
            "group_overlap_count": len(group_overlap),
            "test_groups": len(grouped_test),
            "predictor": "training-majority baseline; held-out groups are unknown",
        },
        "limitations": [
            "The label is deliberately constant within each synthetic group.",
            "This isolates one leakage mechanism and does not measure a real ML model.",
        ],
    }
