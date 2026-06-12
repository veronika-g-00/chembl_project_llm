"""Basic GINE model for molecular pIC50 regression."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool


class BACE1GNN(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.node_projection = nn.Linear(node_feature_dim, hidden_dim)
        self.convolutions = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    edge_dim=edge_feature_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.relu(self.node_projection(x))
        for convolution in self.convolutions:
            x = torch.relu(convolution(x, edge_index, edge_attr))

        graph_embedding = torch.cat(
            [
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )
        return self.regressor(graph_embedding).squeeze(-1)

