import unittest

from prototypes.real_experiments.article08_overfitting import run_real_experiment


class Article08RealExperimentTests(unittest.TestCase):
    def test_public_dataset_experiment_records_disjoint_splits_and_real_provenance(self):
        result = run_real_experiment(seeds=(7,), train_fractions=(0.2, 1.0), max_iter=3)

        self.assertEqual(result["dataset"]["source_id"], "UCI-17-WDBC")
        self.assertEqual(result["evidence_type"], "measured_public_dataset")
        self.assertEqual(result["split_overlap_count"], 0)
        self.assertEqual(len(result["runs"]), 4)
        for run in result["runs"]:
            self.assertGreaterEqual(run["train_auc"], 0.0)
            self.assertLessEqual(run["test_auc"], 1.0)
            self.assertGreater(run["train_rows"], 0)
            self.assertGreater(run["test_rows"], 0)


if __name__ == "__main__":
    unittest.main()
