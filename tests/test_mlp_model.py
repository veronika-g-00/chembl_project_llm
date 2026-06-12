import unittest

import numpy as np
import torch

from libs.mlp_model import BACE1MLP, MLPPreprocessor


class MLPModelTest(unittest.TestCase):
    def test_model_output_shape(self) -> None:
        model = BACE1MLP(input_dim=15, hidden_dims=(8, 4))
        output = model(torch.zeros((3, 15), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3,))

    def test_preprocessor_uses_train_statistics(self) -> None:
        features = np.zeros((3, 6), dtype=np.float32)
        features[:, 4:] = [[1.0, 10.0], [3.0, 14.0], [100.0, 200.0]]
        target = np.asarray([5.0, 7.0, 100.0], dtype=np.float32)
        train_indices = np.asarray([0, 1])

        preprocessor = MLPPreprocessor.fit(
            features,
            target,
            train_indices,
            n_morgan_bits=4,
        )
        transformed = preprocessor.transform_features(features)

        np.testing.assert_allclose(
            transformed[train_indices, 4:].mean(axis=0),
            np.zeros(2),
            atol=1e-6,
        )
        self.assertAlmostEqual(preprocessor.target_mean, 6.0)
        self.assertNotAlmostEqual(float(transformed[2, 4]), 0.0)

    def test_target_transform_is_reversible(self) -> None:
        features = np.zeros((2, 5), dtype=np.float32)
        target = np.asarray([4.0, 8.0], dtype=np.float32)
        preprocessor = MLPPreprocessor.fit(
            features,
            target,
            np.asarray([0, 1]),
            n_morgan_bits=4,
        )
        restored = preprocessor.inverse_target(
            preprocessor.transform_target(target)
        )
        np.testing.assert_allclose(restored, target)


if __name__ == "__main__":
    unittest.main()
