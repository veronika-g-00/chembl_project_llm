import unittest

import numpy as np

from libs.inference import (
    BACE1Predictor,
    activity_label,
    pic50_to_ic50_nm,
)


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = BACE1Predictor()

    def test_predictor_returns_finite_values_and_image(self) -> None:
        result = self.predictor.predict("CCO")
        self.assertTrue(np.isfinite(result.gnn_pic50))
        self.assertTrue(np.isfinite(result.mlp_pic50))
        self.assertGreater(result.gnn_ic50_nm, 0)
        self.assertGreater(result.mlp_ic50_nm, 0)
        self.assertEqual(result.image.size, (600, 420))
        self.assertEqual(len(result.descriptors), 7)

    def test_invalid_smiles_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Niepoprawny SMILES"):
            self.predictor.predict("not-a-smiles")

    def test_pic50_conversion(self) -> None:
        self.assertAlmostEqual(pic50_to_ic50_nm(6.0), 1000.0)
        self.assertEqual(activity_label(8.0), "wysoka przewidywana aktywność")


if __name__ == "__main__":
    unittest.main()
