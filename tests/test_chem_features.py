import unittest

import numpy as np

from libs.chem_features import (
    calculate_descriptors,
    featurize_smiles,
    molecule_from_smiles,
    scaffold_split,
    validate_scaffold_split,
)


class ChemicalFeaturesTest(unittest.TestCase):
    def test_featurization_shape_and_values(self) -> None:
        features, names = featurize_smiles(["CCO", "c1ccccc1"], n_bits=64)
        self.assertEqual(features.shape, (2, 71))
        self.assertEqual(len(names), 71)
        self.assertTrue(np.isfinite(features).all())

    def test_descriptors_have_expected_length(self) -> None:
        descriptors = calculate_descriptors(molecule_from_smiles("CCO"))
        self.assertEqual(descriptors.shape, (7,))

    def test_scaffold_split_has_no_overlap(self) -> None:
        scaffolds = (
            ["A"] * 8
            + ["B"] * 6
            + ["C"] * 4
            + ["D"] * 3
            + ["E"] * 2
            + ["F"] * 2
            + ["G"] * 2
            + ["H"] * 2
            + ["I"] * 2
            + ["J"] * 2
        )
        split = scaffold_split(
            scaffolds,
            train_fraction=0.7,
            validation_fraction=0.15,
            seed=7,
        )
        validate_scaffold_split(scaffolds, split)
        self.assertEqual(
            len(split.train) + len(split.validation) + len(split.test),
            len(scaffolds),
        )


if __name__ == "__main__":
    unittest.main()
