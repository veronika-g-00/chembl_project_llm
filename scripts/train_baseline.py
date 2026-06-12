#!/usr/bin/env python3
"""Train an ExtraTrees BACE1 regression baseline on a scaffold split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from libs.chem_features import (
    DESCRIPTOR_NAMES,
    ScaffoldSplit,
    featurize_smiles,
    scaffold_split,
    validate_scaffold_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Morgan fingerprint ExtraTrees baseline for BACE1."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/bace1_modeling.csv"),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/baseline"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/baseline"),
    )
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_split(
    model: Any,
    features: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    predictions = model.predict(features[indices])
    return regression_metrics(target[indices], predictions)


def grouped_permutation_importance(
    model: Any,
    features: np.ndarray,
    target: np.ndarray,
    n_morgan_bits: int,
    seed: int,
    repeats: int = 5,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    baseline_mae = mean_absolute_error(target, model.predict(features))
    groups: dict[str, np.ndarray] = {
        "morgan_fingerprint": np.arange(n_morgan_bits),
    }
    groups.update(
        {
            name: np.asarray([n_morgan_bits + offset])
            for offset, name in enumerate(DESCRIPTOR_NAMES)
        }
    )

    results: dict[str, dict[str, float]] = {}
    for name, columns in groups.items():
        increases: list[float] = []
        for _ in range(repeats):
            shuffled = features.copy()
            permutation = rng.permutation(len(shuffled))
            shuffled[:, columns] = shuffled[permutation][:, columns]
            permuted_mae = mean_absolute_error(target, model.predict(shuffled))
            increases.append(float(permuted_mae - baseline_mae))
        results[name] = {
            "mae_increase_mean": float(np.mean(increases)),
            "mae_increase_std": float(np.std(increases)),
        }
    return dict(
        sorted(
            results.items(),
            key=lambda item: item[1]["mae_increase_mean"],
            reverse=True,
        )
    )


def split_labels(row_count: int, split: ScaffoldSplit) -> np.ndarray:
    labels = np.empty(row_count, dtype=object)
    labels[split.train] = "train"
    labels[split.validation] = "validation"
    labels[split.test] = "test"
    return labels


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
        title="ExtraTrees: predykcja na scaffold test",
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
        title="Reszty modelu na scaffold test",
        xlabel="Przewidziane pIC50",
        ylabel="Predykcja - wartosc rzeczywista",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Modeling dataset not found: {args.input.resolve()}")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input)
    features, feature_names = featurize_smiles(data["standardized_smiles"])
    target = data["pic50"].to_numpy(dtype=np.float32)
    split = scaffold_split(data["scaffold"], seed=args.seed)
    validate_scaffold_split(data["scaffold"], split)

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(features[split.train], target[split.train])

    model = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(features[split.train], target[split.train])

    metrics = {
        "dummy": {
            "validation": evaluate_split(
                dummy, features, target, split.validation
            ),
            "test": evaluate_split(dummy, features, target, split.test),
        },
        "extra_trees": {
            "train": evaluate_split(model, features, target, split.train),
            "validation": evaluate_split(
                model, features, target, split.validation
            ),
            "test": evaluate_split(model, features, target, split.test),
        },
    }

    importance = grouped_permutation_importance(
        model,
        features[split.test],
        target[split.test],
        n_morgan_bits=2048,
        seed=args.seed,
    )

    labels = split_labels(len(data), split)
    predictions = model.predict(features)
    prediction_table = data[
        [
            "standardized_smiles",
            "scaffold",
            "pic50",
            "n_measurements",
            "high_measurement_variability",
        ]
    ].copy()
    prediction_table["split"] = labels
    prediction_table["prediction"] = predictions
    prediction_table["absolute_error"] = np.abs(
        prediction_table["prediction"] - prediction_table["pic50"]
    )
    prediction_table.to_csv(
        args.reports_dir / "baseline_predictions.csv",
        index=False,
    )

    split_table = data[["standardized_smiles", "scaffold", "pic50"]].copy()
    split_table["split"] = labels
    split_table.to_csv(args.reports_dir / "scaffold_split.csv", index=False)

    metadata = {
        "model": "ExtraTreesRegressor",
        "target": "BACE1 (CHEMBL4822) pIC50 regression",
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "features": {
            "morgan_radius": 2,
            "morgan_bits": 2048,
            "descriptors": DESCRIPTOR_NAMES,
            "total_features": len(feature_names),
        },
        "split": {
            "method": "Bemis-Murcko scaffold split",
            "train_rows": int(len(split.train)),
            "validation_rows": int(len(split.validation)),
            "test_rows": int(len(split.test)),
            "train_scaffolds": int(
                data.iloc[split.train]["scaffold"].nunique()
            ),
            "validation_scaffolds": int(
                data.iloc[split.validation]["scaffold"].nunique()
            ),
            "test_scaffolds": int(
                data.iloc[split.test]["scaffold"].nunique()
            ),
        },
        "metrics": metrics,
        "grouped_permutation_importance": importance,
    }
    (args.reports_dir / "baseline_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "morgan_radius": 2,
            "morgan_bits": 2048,
            "target": metadata["target"],
        },
        args.artifacts_dir / "extra_trees_bace1.joblib",
    )
    plot_predictions(
        prediction_table,
        args.reports_dir / "baseline_predictions.png",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
