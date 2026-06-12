"""Reusable RDKit features and scaffold split utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


DESCRIPTOR_NAMES = [
    "mw",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "heavy_atoms",
]


@dataclass(frozen=True)
class ScaffoldSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def molecule_from_smiles(smiles: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Niepoprawny SMILES: {smiles}")
    return molecule


def standardize_smiles(smiles: str) -> tuple[str, Chem.Mol]:
    molecule = molecule_from_smiles(smiles)
    normalizer = rdMolStandardize.Normalizer()
    largest_fragment = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()

    molecule = normalizer.normalize(molecule)
    molecule = largest_fragment.choose(molecule)
    molecule = uncharger.uncharge(molecule)
    standardized = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    return standardized, molecule


def calculate_descriptors(molecule: Chem.Mol) -> np.ndarray:
    return np.asarray(
        [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            molecule.GetNumHeavyAtoms(),
        ],
        dtype=np.float32,
    )


def calculate_morgan_fingerprint(
    molecule: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    generator = _morgan_generator(radius, n_bits)
    fingerprint = generator.GetFingerprint(molecule)
    values = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values


@lru_cache(maxsize=8)
def _morgan_generator(
    radius: int,
    n_bits: int,
) -> rdFingerprintGenerator.FingerprintGenerator64:
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )


def featurize_smiles(
    smiles_values: Iterable[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    for smiles in smiles_values:
        molecule = molecule_from_smiles(smiles)
        fingerprint = calculate_morgan_fingerprint(
            molecule,
            radius=radius,
            n_bits=n_bits,
        )
        descriptors = calculate_descriptors(molecule)
        rows.append(np.concatenate([fingerprint, descriptors]))

    feature_names = [
        *(f"morgan_{index}" for index in range(n_bits)),
        *DESCRIPTOR_NAMES,
    ]
    return np.vstack(rows).astype(np.float32), feature_names


def bemis_murcko_scaffold(smiles: str) -> str:
    molecule = molecule_from_smiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )
    return scaffold or "[ACYCLIC]"


def scaffold_split(
    scaffolds: Iterable[str],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> ScaffoldSplit:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    scaffold_values = list(scaffolds)
    if not scaffold_values:
        raise ValueError("Cannot split an empty dataset.")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, scaffold in enumerate(scaffold_values):
        groups[scaffold].append(index)

    rng = np.random.default_rng(seed)
    ranked_groups = [
        (indices, float(rng.random())) for indices in groups.values()
    ]
    ranked_groups.sort(key=lambda item: (-len(item[0]), item[1]))

    row_count = len(scaffold_values)
    train_target = int(round(row_count * train_fraction))
    validation_target = int(round(row_count * validation_fraction))

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []

    for indices, _ in ranked_groups:
        if len(train) + len(indices) <= train_target:
            train.extend(indices)
        elif len(validation) + len(indices) <= validation_target:
            validation.extend(indices)
        else:
            test.extend(indices)

    if not train or not validation or not test:
        raise RuntimeError("Scaffold split produced an empty subset.")

    return ScaffoldSplit(
        train=np.asarray(sorted(train), dtype=np.int64),
        validation=np.asarray(sorted(validation), dtype=np.int64),
        test=np.asarray(sorted(test), dtype=np.int64),
    )


def validate_scaffold_split(
    scaffolds: Iterable[str],
    split: ScaffoldSplit,
) -> None:
    scaffold_values = np.asarray(list(scaffolds), dtype=object)
    train_scaffolds = set(scaffold_values[split.train])
    validation_scaffolds = set(scaffold_values[split.validation])
    test_scaffolds = set(scaffold_values[split.test])

    if train_scaffolds & validation_scaffolds:
        raise ValueError("Train and validation contain shared scaffolds.")
    if train_scaffolds & test_scaffolds:
        raise ValueError("Train and test contain shared scaffolds.")
    if validation_scaffolds & test_scaffolds:
        raise ValueError("Validation and test contain shared scaffolds.")

    all_indices = np.concatenate([split.train, split.validation, split.test])
    if len(np.unique(all_indices)) != len(scaffold_values):
        raise ValueError("Split indices do not cover the dataset exactly once.")
