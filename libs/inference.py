"""Inference service used by the Streamlit application."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rdkit.Chem import Draw

from libs.chem_features import (
    DESCRIPTOR_NAMES,
    calculate_descriptors,
    featurize_smiles,
    standardize_smiles,
)
from libs.gnn_model import BACE1GNN
from libs.graph_features import smiles_to_graph
from libs.mlp_model import BACE1MLP, MLPPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GNN_PATH = PROJECT_ROOT / "artifacts/gnn/gnn_bace1.pt"
DEFAULT_MLP_PATH = PROJECT_ROOT / "artifacts/mlp/mlp_bace1.pt"


@dataclass(frozen=True)
class PredictionResult:
    input_smiles: str
    standardized_smiles: str
    gnn_pic50: float
    mlp_pic50: float
    gnn_ic50_nm: float
    mlp_ic50_nm: float
    descriptors: dict[str, float]
    image: Image.Image

    @property
    def model_difference(self) -> float:
        return abs(self.gnn_pic50 - self.mlp_pic50)


def pic50_to_ic50_nm(pic50: float) -> float:
    return 10 ** (9.0 - pic50)


def activity_label(pic50: float) -> str:
    if pic50 >= 8:
        return "wysoka przewidywana aktywność"
    if pic50 >= 6:
        return "umiarkowana przewidywana aktywność"
    return "niska przewidywana aktywność"


class BACE1Predictor:
    def __init__(
        self,
        gnn_path: Path = DEFAULT_GNN_PATH,
        mlp_path: Path = DEFAULT_MLP_PATH,
    ) -> None:
        if not gnn_path.is_file():
            raise FileNotFoundError(f"GNN artifact not found: {gnn_path}")
        if not mlp_path.is_file():
            raise FileNotFoundError(f"MLP artifact not found: {mlp_path}")

        self.gnn_artifact = torch.load(
            gnn_path,
            map_location="cpu",
            weights_only=True,
        )
        self.gnn = BACE1GNN(
            node_feature_dim=self.gnn_artifact["node_feature_dim"],
            edge_feature_dim=self.gnn_artifact["edge_feature_dim"],
            hidden_dim=self.gnn_artifact["hidden_dim"],
            num_layers=self.gnn_artifact["num_layers"],
        )
        self.gnn.load_state_dict(self.gnn_artifact["state_dict"])
        self.gnn.eval()

        self.mlp_artifact = torch.load(
            mlp_path,
            map_location="cpu",
            weights_only=True,
        )
        self.mlp = BACE1MLP(
            input_dim=self.mlp_artifact["input_dim"],
            hidden_dims=tuple(self.mlp_artifact["hidden_dims"]),
        )
        self.mlp.load_state_dict(self.mlp_artifact["state_dict"])
        self.mlp.eval()
        self.mlp_preprocessor = MLPPreprocessor.from_dict(
            self.mlp_artifact["preprocessor"]
        )

    @torch.no_grad()
    def predict(self, smiles: str) -> PredictionResult:
        cleaned_smiles = smiles.strip()
        if not cleaned_smiles:
            raise ValueError("SMILES nie może być pusty.")

        standardized_smiles, molecule = standardize_smiles(cleaned_smiles)

        graph = smiles_to_graph(standardized_smiles)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        normalized_gnn = self.gnn(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            batch,
        )
        gnn_pic50 = float(
            normalized_gnn.item() * self.gnn_artifact["target_std"]
            + self.gnn_artifact["target_mean"]
        )

        raw_features, _ = featurize_smiles(
            [standardized_smiles],
            radius=self.mlp_artifact["morgan_radius"],
            n_bits=self.mlp_artifact["morgan_bits"],
        )
        mlp_features = self.mlp_preprocessor.transform_features(raw_features)
        normalized_mlp = self.mlp(torch.from_numpy(mlp_features)).numpy()
        mlp_pic50 = float(
            self.mlp_preprocessor.inverse_target(normalized_mlp)[0]
        )

        descriptor_values = calculate_descriptors(molecule)
        descriptors = {
            name: float(value)
            for name, value in zip(
                DESCRIPTOR_NAMES,
                descriptor_values,
                strict=True,
            )
        }
        image = Draw.MolToImage(molecule, size=(600, 420))

        values = [gnn_pic50, mlp_pic50, *descriptors.values()]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Model returned a non-finite value.")

        return PredictionResult(
            input_smiles=cleaned_smiles,
            standardized_smiles=standardized_smiles,
            gnn_pic50=gnn_pic50,
            mlp_pic50=mlp_pic50,
            gnn_ic50_nm=pic50_to_ic50_nm(gnn_pic50),
            mlp_ic50_nm=pic50_to_ic50_nm(mlp_pic50),
            descriptors=descriptors,
            image=image,
        )

