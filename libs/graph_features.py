"""RDKit molecular graph featurization for the BACE1 GNN."""

from __future__ import annotations

from typing import TypeVar

import torch
from rdkit import Chem
from torch_geometric.data import Data


T = TypeVar("T")

COMMON_ATOMIC_NUMBERS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
TOTAL_HYDROGENS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def one_hot_with_unknown(value: T, choices: list[T]) -> list[float]:
    encoded = [0.0] * (len(choices) + 1)
    try:
        encoded[choices.index(value)] = 1.0
    except ValueError:
        encoded[-1] = 1.0
    return encoded


def atom_features(atom: Chem.Atom) -> list[float]:
    return [
        *one_hot_with_unknown(
            atom.GetAtomicNum(),
            COMMON_ATOMIC_NUMBERS,
        ),
        *one_hot_with_unknown(atom.GetDegree(), ATOM_DEGREES),
        *one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGES),
        *one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS),
        float(atom.GetIsAromatic()),
        *one_hot_with_unknown(atom.GetTotalNumHs(), TOTAL_HYDROGENS),
        *one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS),
        float(atom.IsInRing()),
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    return [
        *one_hot_with_unknown(bond.GetBondType(), BOND_TYPES),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *one_hot_with_unknown(bond.GetStereo(), BOND_STEREO),
    ]


def feature_dimensions() -> tuple[int, int]:
    molecule = Chem.MolFromSmiles("CC")
    if molecule is None:
        raise RuntimeError("RDKit failed to build the feature reference molecule.")
    return (
        len(atom_features(molecule.GetAtomWithIdx(0))),
        len(bond_features(molecule.GetBondWithIdx(0))),
    )


NODE_FEATURE_DIM, EDGE_FEATURE_DIM = feature_dimensions()


def smiles_to_graph(
    smiles: str,
    target: float | None = None,
    sample_id: int | None = None,
) -> Data:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    node_matrix = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )

    edge_pairs: list[list[int]] = []
    edge_rows: list[list[float]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_pairs.extend([[begin, end], [end, begin]])
        edge_rows.extend([features, features])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_rows, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    graph = Data(
        x=node_matrix,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    if target is not None:
        graph.y = torch.tensor([target], dtype=torch.float32)
    if sample_id is not None:
        graph.sample_id = torch.tensor([sample_id], dtype=torch.long)
    return graph
