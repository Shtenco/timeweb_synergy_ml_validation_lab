import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from prototypes.real_experiments.article08_group_leakage import extract_uci_har_archive


def make_outer_archive(path: Path, nested_members: dict[str, str]) -> None:
    nested_bytes = io.BytesIO()
    with ZipFile(nested_bytes, "w", ZIP_DEFLATED) as inner:
        for name, content in nested_members.items():
            inner.writestr(name, content)
    with ZipFile(path, "w", ZIP_DEFLATED) as outer:
        outer.writestr("UCI HAR Dataset.zip", nested_bytes.getvalue())


class Article08DatasetExtractTests(unittest.TestCase):
    def test_extracts_nested_uci_archive_into_expected_dataset_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "outer.zip"
            members = {
                "UCI HAR Dataset/train/X_train.txt": "1 2 3\n",
                "UCI HAR Dataset/train/y_train.txt": "1\n",
                "UCI HAR Dataset/train/subject_train.txt": "1\n",
                "UCI HAR Dataset/test/X_test.txt": "4 5 6\n",
                "UCI HAR Dataset/test/y_test.txt": "2\n",
                "UCI HAR Dataset/test/subject_test.txt": "2\n",
            }
            make_outer_archive(archive, members)

            extracted = extract_uci_har_archive(archive, root / "data")

            self.assertEqual((extracted / "train" / "X_train.txt").read_text(), "1 2 3\n")

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "outer.zip"
            make_outer_archive(archive, {"UCI HAR Dataset/../../escape.txt": "bad"})

            with self.assertRaises(ValueError):
                extract_uci_har_archive(archive, root / "data")
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
