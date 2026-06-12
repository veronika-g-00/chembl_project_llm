#!/usr/bin/env python3
"""Train the required RDKit graph neural network on the scaffold split."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from libs.gnn_model import BACE1GNN
from libs.graph_features import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    smiles_to_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a basic GINE BACE1 pIC50 regressor."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/bace1_modeling.csv"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("reports/baseline/scaffold_split.csv"),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/gnn"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/gnn"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data_with_split(
    data_path: Path,
    split_path: Path,
) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(f"Modeling data not found: {data_path.resolve()}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Scaffold split not found: {split_path.resolve()}")

    data = pd.read_csv(data_path)
    split_table = pd.read_csv(
        split_path,
        usecols=["standardized_smiles", "split"],
    )
    data = data.merge(
        split_table,
        on="standardized_smiles",
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)
    if data["split"].isna().any():
        raise ValueError("Some molecules are missing from the scaffold split.")
    return data


def build_graphs(
    data: pd.DataFrame,
    target_mean: float,
    target_std: float,
) -> list[Data]:
    graphs: list[Data] = []
    for row_index, row in data.iterrows():
        normalized_target = (float(row["pic50"]) - target_mean) / target_std
        graphs.append(
            smiles_to_graph(
                row["standardized_smiles"],
                target=normalized_target,
                sample_id=int(row_index),
            )
        )
    return graphs


def regression_metrics(
    target: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(mean_squared_error(target, predictions) ** 0.5),
        "r2": float(r2_score(target, predictions)),
    }


@torch.no_grad()
def predict_loader(
    model: BACE1GNN,
    loader: DataLoader,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []

    for batch in loader:
        normalized_prediction = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        predictions.append(
            (
                normalized_prediction.cpu().numpy() * target_std
                + target_mean
            )
        )
        targets.append(batch.y.cpu().numpy() * target_std + target_mean)
        row_indices.append(batch.sample_id.cpu().numpy())

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(row_indices),
    )


def train_model(
    model: BACE1GNN,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    target_mean: float,
    target_std: float,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> tuple[BACE1GNN, list[dict[str, float]], int]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    loss_function = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_validation_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0

        for batch in train_loader:
            optimizer.zero_grad()
            predictions = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            loss = loss_function(predictions, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch.num_graphs
            total_graphs += batch.num_graphs

        validation_predictions, validation_target, _ = predict_loader(
            model,
            validation_loader,
            target_mean,
            target_std,
        )
        validation_mae = float(
            mean_absolute_error(validation_target, validation_predictions)
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse_normalized": total_loss / total_graphs,
                "validation_mae": validation_mae,
            }
        )

        if validation_mae < best_validation_mae - 1e-4:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch


def plot_training(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        history["epoch"],
        history["train_mse_normalized"],
        color="#2A6F97",
    )
    axes[0].set(
        title="GNN: strata treningowa",
        xlabel="Epoka",
        ylabel="MSE znormalizowanego pIC50",
    )

    axes[1].plot(
        history["epoch"],
        history["validation_mae"],
        color="#014F86",
    )
    best_row = history.loc[history["validation_mae"].idxmin()]
    axes[1].scatter(
        [best_row["epoch"]],
        [best_row["validation_mae"]],
        color="#C44536",
        label=f"najlepsza epoka: {int(best_row['epoch'])}",
    )
    axes[1].set(
        title="GNN: jakosc walidacyjna",
        xlabel="Epoka",
        ylabel="MAE pIC50",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(predictions: pd.DataFrame, path: Path) -> None:
    test_data = predictions.loc[predictions["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        test_data["pic50"],
        test_data["prediction"],
        alpha=0.45,
        s=20,
        color="#2A6F97",
    )
    limits = [
        min(test_data["pic50"].min(), test_data["prediction"].min()),
        max(test_data["pic50"].max(), test_data["prediction"].max()),
    ]
    axes[0].plot(limits, limits, linestyle="--", color="#C44536")
    axes[0].set(
        title="GNN: predykcja na scaffold test",
        xlabel="Rzeczywiste pIC50",
        ylabel="Przewidziane pIC50",
    )

    residuals = test_data["prediction"] - test_data["pic50"]
    axes[1].scatter(
        test_data["prediction"],
        residuals,
        alpha=0.45,
        s=20,
        color="#2C7DA0",
    )
    axes[1].axhline(0, linestyle="--", color="#C44536")
    axes[1].set(
        title="GNN: reszty na scaffold test",
        xlabel="Przewidziane pIC50",
        ylabel="Predykcja - wartosc rzeczywista",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs, patience and batch size must be positive.")

    set_seed(args.seed)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    data = load_data_with_split(args.input, args.split)
    train_target = data.loc[data["split"] == "train", "pic50"]
    target_mean = float(train_target.mean())
    target_std = float(train_target.std(ddof=0))
    if target_std < 1e-8:
        target_std = 1.0

    graphs = build_graphs(data, target_mean, target_std)
    split_graphs = {
        split_name: [
            graph
            for graph, split_value in zip(graphs, data["split"], strict=True)
            if split_value == split_name
        ]
        for split_name in ("train", "validation", "test")
    }

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        split_graphs["train"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    evaluation_loaders = {
        split_name: DataLoader(
            split_graphs[split_name],
            batch_size=args.batch_size,
            shuffle=False,
        )
        for split_name in ("train", "validation", "test")
    }

    model = BACE1GNN(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=3,
    )
    model, history, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=evaluation_loaders["validation"],
        target_mean=target_mean,
        target_std=target_std,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    all_predictions = np.empty(len(data), dtype=np.float32)
    metrics: dict[str, dict[str, float]] = {}
    for split_name, loader in evaluation_loaders.items():
        predictions, target, row_indices = predict_loader(
            model,
            loader,
            target_mean,
            target_std,
        )
        all_predictions[row_indices] = predictions
        metrics[split_name] = regression_metrics(target, predictions)

    prediction_table = data[
        [
            "standardized_smiles",
            "scaffold",
            "pic50",
            "n_measurements",
            "high_measurement_variability",
            "split",
        ]
    ].copy()
    prediction_table["prediction"] = all_predictions
    prediction_table["absolute_error"] = np.abs(
        prediction_table["prediction"] - prediction_table["pic50"]
    )
    prediction_table.to_csv(
        args.reports_dir / "gnn_predictions.csv",
        index=False,
    )

    history_table = pd.DataFrame(history)
    history_table.to_csv(
        args.reports_dir / "gnn_training_history.csv",
        index=False,
    )

    metadata: dict[str, Any] = {
        "model": "BACE1GNN with GINEConv",
        "target": "BACE1 (CHEMBL4822) pIC50 regression",
        "architecture": {
            "node_feature_dim": NODE_FEATURE_DIM,
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
            "graph_layers": 3,
            "pooling": "concatenated global mean and max",
            "batch_norm": False,
            "dropout": False,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-5,
            "loss": "MSE on standardized pIC50",
            "batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "early_stopping_patience": args.patience,
            "seed": args.seed,
            "device": "cpu",
        },
        "features": {
            "source": "RDKit molecular graph",
            "node_features": [
                "atomic number",
                "degree",
                "formal charge",
                "hybridization",
                "aromaticity",
                "total hydrogens",
                "chirality",
                "ring membership",
            ],
            "edge_features": [
                "bond type",
                "conjugation",
                "ring membership",
                "stereochemistry",
            ],
            "bidirectional_edges": True,
        },
        "target_scaling": {
            "fit_on": "train only",
            "mean": target_mean,
            "std": target_std,
        },
        "split": {
            name: int(len(values))
            for name, values in split_graphs.items()
        },
        "metrics": metrics,
    }
    (args.reports_dir / "gnn_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "node_feature_dim": NODE_FEATURE_DIM,
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
            "num_layers": 3,
            "target_mean": target_mean,
            "target_std": target_std,
            "target": metadata["target"],
            "features": metadata["features"],
            "metrics": metrics,
        },
        args.artifacts_dir / "gnn_bace1.pt",
    )
    plot_training(
        history_table,
        args.reports_dir / "gnn_training.png",
    )
    plot_predictions(
        prediction_table,
        args.reports_dir / "gnn_predictions.png",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
