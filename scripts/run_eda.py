#!/usr/bin/env python3
"""Run reproducible EDA and prepare the aggregated BACE1 modeling dataset."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


REQUIRED_COLUMNS = {
    "activity_id",
    "assay_id",
    "doc_id",
    "molregno",
    "canonical_smiles",
    "pic50",
    "confidence_score",
    "target_chembl_id",
    "target_organism",
}

DESCRIPTOR_COLUMNS = [
    "mw",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "heavy_atoms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore and aggregate the strict ChEMBL 36 BACE1 dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/bace1_chembl36_strict.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/bace1_modeling.csv"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {path.resolve()}")

    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Input dataset is missing columns: {sorted(missing_columns)}"
        )
    return data


def standardize_molecule(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"valid_smiles": False}

    normalizer = rdMolStandardize.Normalizer()
    largest_fragment = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()

    molecule = normalizer.normalize(molecule)
    molecule = largest_fragment.choose(molecule)
    molecule = uncharger.uncharge(molecule)

    standardized_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )

    return {
        "valid_smiles": True,
        "standardized_smiles": standardized_smiles,
        "structure_changed": standardized_smiles != smiles,
        "scaffold": scaffold or "[ACYCLIC]",
        "mw": Descriptors.MolWt(molecule),
        "logp": Crippen.MolLogP(molecule),
        "tpsa": rdMolDescriptors.CalcTPSA(molecule),
        "hbd": Lipinski.NumHDonors(molecule),
        "hba": Lipinski.NumHAcceptors(molecule),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
    }


def add_chemical_features(data: pd.DataFrame) -> pd.DataFrame:
    unique_smiles = data["canonical_smiles"].drop_duplicates()
    feature_cache = {
        smiles: standardize_molecule(smiles) for smiles in unique_smiles
    }
    features = pd.DataFrame(
        [feature_cache[smiles] for smiles in data["canonical_smiles"]],
        index=data.index,
    )
    return pd.concat([data, features], axis=1)


def aggregate_measurements(data: pd.DataFrame) -> pd.DataFrame:
    valid_data = data.loc[data["valid_smiles"]].copy()

    structure_features = (
        valid_data[
            ["standardized_smiles", "scaffold", *DESCRIPTOR_COLUMNS]
        ]
        .drop_duplicates("standardized_smiles")
        .set_index("standardized_smiles")
    )

    aggregated = valid_data.groupby("standardized_smiles").agg(
        pic50=("pic50", "median"),
        n_measurements=("pic50", "size"),
        n_assays=("assay_id", "nunique"),
        n_documents=("doc_id", "nunique"),
        n_molregno=("molregno", "nunique"),
        pic50_min=("pic50", "min"),
        pic50_max=("pic50", "max"),
        pic50_std=("pic50", "std"),
        target_chembl_id=("target_chembl_id", "first"),
        target_organism=("target_organism", "first"),
    )
    aggregated["pic50_range"] = (
        aggregated["pic50_max"] - aggregated["pic50_min"]
    )
    aggregated["high_measurement_variability"] = (
        aggregated["pic50_range"] > 1.0
    )
    aggregated = aggregated.join(structure_features).reset_index()
    aggregated["pic50_std"] = aggregated["pic50_std"].fillna(0.0)

    ordered_columns = [
        "standardized_smiles",
        "pic50",
        "n_measurements",
        "n_assays",
        "n_documents",
        "n_molregno",
        "pic50_min",
        "pic50_max",
        "pic50_std",
        "pic50_range",
        "high_measurement_variability",
        "scaffold",
        *DESCRIPTOR_COLUMNS,
        "target_chembl_id",
        "target_organism",
    ]
    return aggregated[ordered_columns].sort_values(
        "standardized_smiles"
    ).reset_index(drop=True)


def iqr_outlier_summary(series: pd.Series) -> dict[str, Any]:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(outlier_mask.sum()),
    }


def build_summary(raw: pd.DataFrame, featured: pd.DataFrame, modeling: pd.DataFrame) -> dict[str, Any]:
    repeated = modeling.loc[modeling["n_measurements"] > 1]
    correlations = modeling[[*DESCRIPTOR_COLUMNS, "pic50"]].corr(
        method="pearson"
    )

    return {
        "software": {
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "raw_data": {
            "rows": int(len(raw)),
            "columns": int(raw.shape[1]),
            "missing_values": {
                column: int(count)
                for column, count in raw.isna().sum().items()
            },
            "exact_duplicate_rows": int(raw.duplicated().sum()),
            "duplicate_activity_ids": int(
                raw["activity_id"].duplicated().sum()
            ),
            "unique_raw_smiles": int(raw["canonical_smiles"].nunique()),
            "pic50": {
                key: float(value)
                for key, value in raw["pic50"].describe().items()
            },
            "pic50_iqr_outliers": iqr_outlier_summary(raw["pic50"]),
        },
        "chemical_quality": {
            "invalid_smiles": int((~featured["valid_smiles"]).sum()),
            "rows_changed_by_standardization": int(
                featured["structure_changed"].fillna(False).sum()
            ),
            "unique_standardized_smiles": int(
                modeling["standardized_smiles"].nunique()
            ),
            "unique_scaffolds": int(modeling["scaffold"].nunique()),
            "acyclic_structures": int(
                (modeling["scaffold"] == "[ACYCLIC]").sum()
            ),
            "largest_scaffold_group": int(
                modeling["scaffold"].value_counts().max()
            ),
            "molecules_mw_above_1000": int((modeling["mw"] > 1000).sum()),
        },
        "repeated_measurements": {
            "structures_with_repeats": int(len(repeated)),
            "structures_with_pic50_range_above_0_5": int(
                (repeated["pic50_range"] > 0.5).sum()
            ),
            "structures_with_pic50_range_above_1_0": int(
                (repeated["pic50_range"] > 1.0).sum()
            ),
            "structures_with_pic50_range_above_2_0": int(
                (repeated["pic50_range"] > 2.0).sum()
            ),
            "median_pic50_range": float(repeated["pic50_range"].median()),
            "maximum_pic50_range": float(repeated["pic50_range"].max()),
        },
        "modeling_data": {
            "rows": int(len(modeling)),
            "pic50": {
                key: float(value)
                for key, value in modeling["pic50"].describe().items()
            },
            "high_measurement_variability": int(
                modeling["high_measurement_variability"].sum()
            ),
        },
        "descriptor_pic50_correlations": {
            descriptor: float(correlations.loc[descriptor, "pic50"])
            for descriptor in DESCRIPTOR_COLUMNS
        },
    }


def save_profile(raw: pd.DataFrame, modeling: pd.DataFrame, path: Path) -> None:
    buffer = io.StringIO()
    buffer.write("RAW DATA INFO\n")
    buffer.write("=" * 80 + "\n")
    raw.info(buf=buffer)
    buffer.write("\n\nRAW DATA DESCRIBE\n")
    buffer.write("=" * 80 + "\n")
    buffer.write(raw.describe(include="all").to_string())
    buffer.write("\n\nMODELING DATA INFO\n")
    buffer.write("=" * 80 + "\n")
    modeling.info(buf=buffer)
    buffer.write("\n\nMODELING DATA DESCRIBE\n")
    buffer.write("=" * 80 + "\n")
    buffer.write(modeling.describe(include="all").to_string())
    buffer.write("\n")
    path.write_text(buffer.getvalue(), encoding="utf-8")


def plot_pic50_distribution(raw: pd.DataFrame, modeling: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(raw["pic50"], bins=40, kde=True, ax=axes[0], color="#2A6F97")
    axes[0].axvline(raw["pic50"].median(), color="#C44536", linestyle="--")
    axes[0].set(
        title="Rozklad pIC50 przed agregacja",
        xlabel="pIC50",
        ylabel="Liczba pomiarow",
    )

    sns.boxplot(x=modeling["pic50"], ax=axes[1], color="#61A5C2")
    axes[1].set(
        title="pIC50 po agregacji mediana",
        xlabel="pIC50",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_repeat_variability(modeling: pd.DataFrame, path: Path) -> None:
    repeated = modeling.loc[modeling["n_measurements"] > 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(
        repeated["pic50_range"],
        bins=35,
        ax=axes[0],
        color="#2C7DA0",
    )
    axes[0].axvline(1.0, color="#C44536", linestyle="--", label="prog 1.0")
    axes[0].set(
        title="Rozstęp powtorzonych pomiarow",
        xlabel="max(pIC50) - min(pIC50)",
        ylabel="Liczba struktur",
    )
    axes[0].legend()

    sns.scatterplot(
        data=repeated,
        x="n_measurements",
        y="pic50_range",
        alpha=0.45,
        s=28,
        ax=axes[1],
        color="#014F86",
    )
    axes[1].axhline(1.0, color="#C44536", linestyle="--")
    axes[1].set(
        title="Liczba pomiarow a ich rozbieznosc",
        xlabel="Liczba pomiarow struktury",
        ylabel="Rozstęp pIC50",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(modeling: pd.DataFrame, path: Path) -> None:
    columns = [*DESCRIPTOR_COLUMNS, "pic50"]
    correlation = modeling[columns].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Korelacje deskryptorow RDKit i pIC50")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure_directory = args.report_dir / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    raw_data = load_data(args.input)
    featured_data = add_chemical_features(raw_data)
    modeling_data = aggregate_measurements(featured_data)

    modeling_data.to_csv(args.output, index=False)
    summary = build_summary(raw_data, featured_data, modeling_data)
    (args.report_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    save_profile(
        raw_data,
        modeling_data,
        args.report_dir / "eda_table_profile.txt",
    )
    plot_pic50_distribution(
        raw_data,
        modeling_data,
        figure_directory / "01_pic50_distribution.png",
    )
    plot_repeat_variability(
        modeling_data,
        figure_directory / "02_repeat_variability.png",
    )
    plot_correlation_heatmap(
        modeling_data,
        figure_directory / "03_descriptor_correlations.png",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
