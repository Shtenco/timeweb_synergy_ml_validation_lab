import unittest

import numpy as np

from prototypes.real_experiments.article08_group_leakage import compare_splits


class Article08GroupLeakageTests(unittest.TestCase):
    def test_row_split_mixes_subjects_but_official_subject_split_does_not(self):
        X = np.array([[float(i), float(i % 3)] for i in range(24)])
        y = np.array([i % 2 for i in range(24)])
        subjects = np.repeat(np.arange(6), 4)

        result = compare_splits(
            X,
            y,
            subjects,
            official_train_subjects={0, 1, 2, 3},
            official_test_subjects={4, 5},
            seeds=(4,),
            tree_count=3,
        )

        row_split, subject_split = result["runs"]
        self.assertGreater(row_split["subject_overlap_count"], 0)
        self.assertEqual(subject_split["subject_overlap_count"], 0)
        self.assertEqual(result["evidence_type"], "measured_public_dataset")


if __name__ == "__main__":
    unittest.main()
