"""
Adaptive Multi-Omics Fusion module.
Combines molecular embedding + clinical features + treatment features
using attention-based adaptive fusion.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _check_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError("PyTorch required: pip install torch")


def _build_fusion_module(
    molecular_dim, clinical_dim, treatment_dim, fusion_dim, dropout
):
    torch, nn = _check_torch()

    class AdaptiveFusionModule(nn.Module):
        """
        Attention-weighted fusion of three modalities:
        - Molecular embedding (from GCN/GAT)
        - Clinical features (encoded)
        - Treatment features (encoded)

        Uses learned attention weights to adaptively combine modalities.
        """

        def __init__(self):
            super().__init__()
            # Project each modality to fusion_dim
            self.mol_proj = nn.Sequential(
                nn.Linear(molecular_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.clin_proj = nn.Sequential(
                nn.Linear(clinical_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.treat_proj = nn.Sequential(
                nn.Linear(treatment_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Attention scores (one per modality)
            self.attention = nn.Sequential(
                nn.Linear(3 * fusion_dim, 3),
                nn.Softmax(dim=1),
            )

            # Final fusion MLP
            self.output_layer = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, mol_emb, clin_feat, treat_feat):
            # Project each modality
            m = self.mol_proj(mol_emb)    # (B, fusion_dim)
            c = self.clin_proj(clin_feat)  # (B, fusion_dim)
            t = self.treat_proj(treat_feat)  # (B, fusion_dim)

            # Compute attention weights
            concat = torch.cat([m, c, t], dim=1)  # (B, 3*fusion_dim)
            attn_weights = self.attention(concat)   # (B, 3)

            # Weighted sum
            fused = (
                attn_weights[:, 0:1] * m
                + attn_weights[:, 1:2] * c
                + attn_weights[:, 2:3] * t
            )  # (B, fusion_dim)

            out = self.output_layer(fused)
            return out, attn_weights

    return AdaptiveFusionModule()


class AdaptiveFusion:
    """Wrapper for the adaptive fusion module."""

    def __init__(
        self,
        molecular_dim: int = 256,
        clinical_dim: int = 64,
        treatment_dim: int = 32,
        fusion_dim: int = 128,
        dropout: float = 0.3,
        device: str = "auto",
    ):
        torch, nn = _check_torch()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.module = _build_fusion_module(
            molecular_dim=molecular_dim,
            clinical_dim=clinical_dim,
            treatment_dim=treatment_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        ).to(self.device)

        logger.info(
            f"AdaptiveFusion initialized: mol={molecular_dim}, "
            f"clin={clinical_dim}, treat={treatment_dim} → {fusion_dim}"
        )

    def get_module(self):
        return self.module
