"""MLP model and preprocessing shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class MLPPreprocessor:
    descriptor_mean: np.ndarray
    descriptor_std: np.ndarray
    target_mean: float
    target_std: float
    n_morgan_bits: int = 2048

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        target: np.ndarray,
        train_indices: np.ndarray,
        n_morgan_bits: int = 2048,
    ) -> "MLPPreprocessor":
        descriptors = features[train_indices, n_morgan_bits:]
        descriptor_mean = descriptors.mean(axis=0).astype(np.float32)
        descriptor_std = descriptors.std(axis=0).astype(np.float32)
        descriptor_std[descriptor_std < 1e-8] = 1.0

        train_target = target[train_indices]
        target_mean = float(train_target.mean())
        target_std = float(train_target.std())
        if target_std < 1e-8:
            target_std = 1.0

        return cls(
            descriptor_mean=descriptor_mean,
            descriptor_std=descriptor_std,
            target_mean=target_mean,
            target_std=target_std,
            n_morgan_bits=n_morgan_bits,
        )

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        transformed = features.astype(np.float32, copy=True)
        transformed[:, self.n_morgan_bits :] = (
            transformed[:, self.n_morgan_bits :] - self.descriptor_mean
        ) / self.descriptor_std
        return transformed

    def transform_target(self, target: np.ndarray) -> np.ndarray:
        return ((target - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_target(self, normalized_target: np.ndarray) -> np.ndarray:
        return normalized_target * self.target_std + self.target_mean

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor_mean": self.descriptor_mean.tolist(),
            "descriptor_std": self.descriptor_std.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "n_morgan_bits": self.n_morgan_bits,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "MLPPreprocessor":
        return cls(
            descriptor_mean=np.asarray(
                values["descriptor_mean"],
                dtype=np.float32,
            ),
            descriptor_std=np.asarray(
                values["descriptor_std"],
                dtype=np.float32,
            ),
            target_mean=float(values["target_mean"]),
            target_std=float(values["target_std"]),
            n_morgan_bits=int(values["n_morgan_bits"]),
        )


class BACE1MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2055,
        hidden_dims: tuple[int, int] = (256, 128),
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

