"""
Graph Convolutional Network (GCN) model for molecular representation learning.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _check_torch():
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv, global_mean_pool
        return torch, nn, GCNConv, global_mean_pool
    except ImportError:
        raise ImportError(
            "PyTorch and torch-geometric are required.\n"
            "Install: pip install torch torch-geometric"
        )


class GCNModel:
    """
    Graph Convolutional Network for patient-level molecular embedding.

    Architecture:
        Input node features (n_genes x 1)
        → GCN Layer 1 → BatchNorm → ReLU → Dropout
        → GCN Layer 2 → BatchNorm → ReLU → Dropout
        → Global Mean Pooling
        → Patient-level embedding (hidden_dim,)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: str = "auto",
    ):
        torch, nn, GCNConv, global_mean_pool = _check_torch()

        self.device_str = device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = _GCNModule(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        logger.info(
            f"GCN initialized: {in_channels}→{hidden_dim}×{num_layers} "
            f"on {self.device}"
        )

    def get_module(self):
        return self.model

    def embed(self, data_list: list) -> np.ndarray:
        """Generate patient-level embeddings for a list of PyG Data objects."""
        import torch
        from torch_geometric.loader import DataLoader

        self.model.eval()
        loader = DataLoader(data_list, batch_size=32, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                emb = self.model(batch.x, batch.edge_index, batch.batch)
                embeddings.append(emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)


def _build_gcn_module(in_channels, hidden_dim, num_layers, dropout):
    """Build the PyTorch nn.Module for GCN."""
    torch, nn, GCNConv, global_mean_pool = _check_torch()

    class GCNModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.dropout = nn.Dropout(dropout)

            dims = [in_channels] + [hidden_dim] * num_layers
            for i in range(num_layers):
                self.convs.append(GCNConv(dims[i], dims[i + 1]))
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        def forward(self, x, edge_index, batch):
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = torch.relu(x)
                x = self.dropout(x)
            # Global mean pooling → patient-level embedding
            out = global_mean_pool(x, batch)
            return out

    return GCNModule()


class _GCNModule:
    """Lazy wrapper that builds the PyTorch module only when torch is available."""
    def __new__(cls, in_channels, hidden_dim, num_layers, dropout):
        return _build_gcn_module(in_channels, hidden_dim, num_layers, dropout)
