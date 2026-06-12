import tempfile
import unittest
from pathlib import Path

from scripts.extract_bace1 import (
    TARGET_CHEMBL_ID,
    batched,
    connect_read_only,
    extract_rows,
    load_target_assays,
    validate_database,
    validate_limit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "chembl_36.db"


class ExtractionHelpersTest(unittest.TestCase):
    def test_batched_preserves_order(self) -> None:
        self.assertEqual(batched([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_limit_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            validate_limit(0)


@unittest.skipUnless(DATABASE_PATH.is_file(), "ChEMBL 36 database is unavailable")
class ExtractionIntegrationTest(unittest.TestCase):
    def test_extracts_strict_bace1_sample(self) -> None:
        connection = connect_read_only(DATABASE_PATH)
        try:
            database_info = validate_database(connection)
            target = database_info["target"]
            assays = load_target_assays(connection, target["tid"])

            with tempfile.TemporaryDirectory() as temporary_directory:
                output = Path(temporary_directory) / "bace1_sample.csv"
                statistics = extract_rows(
                    connection,
                    output,
                    limit=10,
                    target=target,
                    assays=assays,
                )

                self.assertEqual(target["chembl_id"], TARGET_CHEMBL_ID)
                self.assertEqual(statistics["records"], 10)
                self.assertGreater(statistics["unique_smiles"], 0)
                self.assertTrue(output.is_file())
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
