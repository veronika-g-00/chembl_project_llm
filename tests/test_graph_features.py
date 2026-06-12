import unittest
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Batch

from libs.gnn_model import BACE1GNN
from libs.graph_features import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    smiles_to_graph,
)
from scripts.train_gnn import load_data_with_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GraphFeaturesTest(unittest.TestCase):
    def test_ethanol_graph_has_bidirectional_edges(self) -> None:
        graph = smiles_to_graph("CCO", target=6.0, sample_id=4)

        self.assertEqual(tuple(graph.x.shape), (3, NODE_FEATURE_DIM))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 4))
        self.assertEqual(tuple(graph.edge_attr.shape), (4, EDGE_FEATURE_DIM))
        self.assertEqual(float(graph.y.item()), 6.0)
        self.assertEqual(int(graph.sample_id.item()), 4)

        edges = {tuple(edge) for edge in graph.edge_index.t().tolist()}
        for begin, end in edges:
            self.assertIn((end, begin), edges)

    def test_single_atom_graph_has_empty_edges(self) -> None:
        graph = smiles_to_graph("[Na+]")
        self.assertEqual(tuple(graph.edge_index.shape), (2, 0))
        self.assertEqual(tuple(graph.edge_attr.shape), (0, EDGE_FEATURE_DIM))

    def test_gnn_returns_one_value_per_graph(self) -> None:
        batch = Batch.from_data_list(
            [smiles_to_graph("CCO"), smiles_to_graph("c1ccccc1")]
        )
        model = BACE1GNN(
            node_feature_dim=NODE_FEATURE_DIM,
            edge_feature_dim=EDGE_FEATURE_DIM,
            hidden_dim=16,
            num_layers=2,
        )
        output = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.isfinite(output).all())

    def test_batch_preserves_sample_ids(self) -> None:
        batch = Batch.from_data_list(
            [
                smiles_to_graph("CCO", sample_id=4),
                smiles_to_graph("c1ccccc1", sample_id=9),
            ]
        )
        self.assertEqual(batch.sample_id.tolist(), [4, 9])

    @unittest.skipUnless(
        (PROJECT_ROOT / "reports/baseline/scaffold_split.csv").is_file(),
        "Scaffold split is unavailable",
    )
    def test_loaded_split_has_contiguous_index(self) -> None:
        data = load_data_with_split(
            PROJECT_ROOT / "data/processed/bace1_modeling.csv",
            PROJECT_ROOT / "reports/baseline/scaffold_split.csv",
        )
        self.assertEqual(
            data.index.tolist(),
            list(range(len(data))),
        )
        self.assertEqual(
            data["split"].value_counts().to_dict(),
            {"train": 6440, "validation": 805, "test": 805},
        )


if __name__ == "__main__":
    unittest.main()
