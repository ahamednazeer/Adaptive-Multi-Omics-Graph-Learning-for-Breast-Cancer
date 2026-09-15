"""
Graph Attention Network (GAT) model for molecular representation learning.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _check_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv, global_mean_pool
        return torch, nn, F, GATConv, global_mean_pool
    except ImportError:
        raise ImportError(
            "PyTorch and torch-geometric required: pip install torch torch-geometric"
        )


def _build_gat_module(in_channels, hidden_dim, num_heads, dropout, attention_dropout):
    torch, nn, F, GATConv, global_mean_pool = _check_torch()

    class GATModule(nn.Module):
        def __init__(self):
            super().__init__()
            # Layer 1: in_channels → hidden_dim (multi-head)
            self.conv1 = GATConv(
                in_channels,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=attention_dropout,
                concat=True,  # output = num_heads * (hidden_dim // num_heads) = hidden_dim
            )
            self.bn1 = nn.BatchNorm1d(hidden_dim)

            # Layer 2: hidden_dim → hidden_dim (single head for final embedding)
            self.conv2 = GATConv(
                hidden_dim,
                hidden_dim,
                heads=1,
                dropout=attention_dropout,
                concat=False,
            )
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, batch):
            # Layer 1
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.elu(x)
            x = self.dropout(x)

            # Layer 2
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.dropout(x)

            # Global mean pooling → patient embedding
            return global_mean_pool(x, batch)

    return GATModule()


class GATModel:
    """
    Graph Attention Network for patient-level molecular embedding.

    Compared to GCN, GAT uses attention weights on edges — letting
    the model learn which gene-gene interactions are most important
    for pCR prediction.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
        attention_dropout: float = 0.1,
        device: str = "auto",
    ):
        torch, nn, F, GATConv, global_mean_pool = _check_torch()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = _build_gat_module(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        ).to(self.device)

        logger.info(
            f"GAT initialized: {in_channels}→{hidden_dim} "
            f"({num_heads} heads) on {self.device}"
        )

    def get_module(self):
        return self.model

    def embed(self, data_list: list) -> np.ndarray:
        """Generate patient-level embeddings."""
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
