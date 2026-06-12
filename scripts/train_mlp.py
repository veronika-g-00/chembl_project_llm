#!/usr/bin/env python3
"""Train the required MLP BACE1 regressor on the scaffold split."""

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
from torch.utils.data import DataLoader, TensorDataset

from libs.chem_features import DESCRIPTOR_NAMES, featurize_smiles
from libs.mlp_model import BACE1MLP, MLPPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLP BACE1 pIC50 regressor."
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
        default=Path("artifacts/mlp"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/mlp"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_split_data(
    data_path: Path,
    split_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Modeling data not found: {data_path.resolve()}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Scaffold split not found: {split_path.resolve()}")

    data = pd.read_csv(data_path)
    split_table = pd.read_csv(
        split_path,
        usecols=["standardized_smiles", "split"],
    )
    if split_table["standardized_smiles"].duplicated().any():
        raise ValueError("Split table contains duplicated SMILES.")

    data = data.merge(
        split_table,
        on="standardized_smiles",
        how="left",
        validate="one_to_one",
    )
    if data["split"].isna().any():
        raise ValueError("Some molecules are missing from the scaffold split.")

    indices = {
        name: np.flatnonzero(data["split"].to_numpy() == name)
        for name in ("train", "validation", "test")
    }
    if any(len(values) == 0 for values in indices.values()):
        raise ValueError("Scaffold split contains an empty subset.")
    return data, indices


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
def predict(
    model: nn.Module,
    features: np.ndarray,
    preprocessor: MLPPreprocessor,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(features))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    normalized_predictions: list[np.ndarray] = []
    for (batch_features,) in loader:
        normalized_predictions.append(
            model(batch_features).cpu().numpy()
        )
    return preprocessor.inverse_target(
        np.concatenate(normalized_predictions)
    )


def train_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    preprocessor: MLPPreprocessor,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]], int]:
    generator = torch.Generator().manual_seed(seed)
    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_target),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

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
        total_rows = 0

        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_features)
            total_rows += len(batch_features)

        validation_predictions = predict(
            model,
            validation_features,
            preprocessor,
        )
        validation_original_target = preprocessor.inverse_target(
            validation_target
        )
        validation_mae = float(
            mean_absolute_error(
                validation_original_target,
                validation_predictions,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse_normalized": total_loss / total_rows,
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
        title="MLP: strata treningowa",
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
        title="MLP: jakosc walidacyjna",
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
        title="MLP: predykcja na scaffold test",
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
        title="MLP: reszty na scaffold test",
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

    data, split_indices = load_split_data(args.input, args.split)
    raw_features, feature_names = featurize_smiles(
        data["standardized_smiles"]
    )
    target = data["pic50"].to_numpy(dtype=np.float32)

    preprocessor = MLPPreprocessor.fit(
        raw_features,
        target,
        split_indices["train"],
    )
    features = preprocessor.transform_features(raw_features)
    normalized_target = preprocessor.transform_target(target)

    model = BACE1MLP(input_dim=features.shape[1])
    model, history, best_epoch = train_model(
        model=model,
        train_features=features[split_indices["train"]],
        train_target=normalized_target[split_indices["train"]],
        validation_features=features[split_indices["validation"]],
        validation_target=normalized_target[split_indices["validation"]],
        preprocessor=preprocessor,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    all_predictions = predict(model, features, preprocessor)
    metrics = {
        split_name: regression_metrics(
            target[indices],
            all_predictions[indices],
        )
        for split_name, indices in split_indices.items()
    }

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
        args.reports_dir / "mlp_predictions.csv",
        index=False,
    )

    history_table = pd.DataFrame(history)
    history_table.to_csv(
        args.reports_dir / "mlp_training_history.csv",
        index=False,
    )

    metadata: dict[str, Any] = {
        "model": "BACE1MLP",
        "target": "BACE1 (CHEMBL4822) pIC50 regression",
        "architecture": {
            "input_dim": int(features.shape[1]),
            "hidden_dims": [256, 128],
            "activation": "ReLU",
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
            "morgan_radius": 2,
            "morgan_bits": 2048,
            "descriptors": DESCRIPTOR_NAMES,
            "descriptor_scaling_fit_on": "train only",
            "total_features": len(feature_names),
        },
        "split": {
            name: int(len(indices))
            for name, indices in split_indices.items()
        },
        "metrics": metrics,
    }
    (args.reports_dir / "mlp_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(features.shape[1]),
            "hidden_dims": [256, 128],
            "preprocessor": preprocessor.to_dict(),
            "morgan_radius": 2,
            "morgan_bits": 2048,
            "descriptor_names": DESCRIPTOR_NAMES,
            "target": metadata["target"],
            "metrics": metrics,
        },
        args.artifacts_dir / "mlp_bace1.pt",
    )
    plot_training(
        history_table,
        args.reports_dir / "mlp_training.png",
    )
    plot_predictions(
        prediction_table,
        args.reports_dir / "mlp_predictions.png",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

