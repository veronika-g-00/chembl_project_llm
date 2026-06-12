import unittest

import pandas as pd

from scripts.run_eda import aggregate_measurements, iqr_outlier_summary


class EdaTest(unittest.TestCase):
    def test_aggregation_uses_median_and_preserves_variability(self) -> None:
        data = pd.DataFrame(
            {
                "valid_smiles": [True, True, True],
                "standardized_smiles": ["CCO", "CCO", "CCN"],
                "pic50": [5.0, 7.0, 6.0],
                "assay_id": [1, 2, 3],
                "doc_id": [10, 20, 30],
                "molregno": [100, 100, 200],
                "target_chembl_id": ["CHEMBL4822"] * 3,
                "target_organism": ["Homo sapiens"] * 3,
                "scaffold": ["[ACYCLIC]"] * 3,
                "mw": [46.1, 46.1, 45.1],
                "logp": [-0.1, -0.1, -0.2],
                "tpsa": [20.2, 20.2, 26.0],
                "hbd": [1, 1, 1],
                "hba": [1, 1, 1],
                "rotatable_bonds": [0, 0, 0],
                "heavy_atoms": [3, 3, 3],
            }
        )

        result = aggregate_measurements(data).set_index("standardized_smiles")

        self.assertEqual(result.loc["CCO", "pic50"], 6.0)
        self.assertEqual(result.loc["CCO", "n_measurements"], 2)
        self.assertEqual(result.loc["CCO", "pic50_range"], 2.0)
        self.assertTrue(result.loc["CCO", "high_measurement_variability"])

    def test_iqr_summary_counts_extreme_values(self) -> None:
        series = pd.Series([1.0, 2.0, 2.0, 3.0, 100.0])
        summary = iqr_outlier_summary(series)
        self.assertEqual(summary["outlier_count"], 1)


if __name__ == "__main__":
    unittest.main()
