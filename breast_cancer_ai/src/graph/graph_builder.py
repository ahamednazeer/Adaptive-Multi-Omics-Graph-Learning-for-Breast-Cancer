"""
Graph builder — constructs PyTorch Geometric Data objects from STRING network
and patient omics features.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_networkx_graph(
    string_df: pd.DataFrame,
    gene_list: list[str],
) -> "nx.Graph":
    """
    Build a NetworkX undirected graph from STRING interactions.

    Args:
        string_df: STRING interactions DataFrame (gene1, gene2, combined_score)
        gene_list: List of gene/protein names to include as nodes

    Returns:
        NetworkX Graph
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    G = nx.Graph()
    G.add_nodes_from(gene_list)

    for _, row in string_df.iterrows():
        g1, g2, score = row["gene1"], row["gene2"], row["combined_score"]
        if g1 in G and g2 in G:
            # Normalize score to [0, 1]
            weight = float(score) / 1000.0
            G.add_edge(g1, g2, weight=weight)

    logger.info(
        f"Graph built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def build_pyg_data(
    patient_features: pd.DataFrame,
    gene_list: list[str],
    string_df: pd.DataFrame,
    labels: Optional[pd.Series] = None,
) -> list:
    """
    Build a list of PyTorch Geometric Data objects — one per patient.

    Each graph has:
    - x: node feature matrix (n_genes x n_patients... actually per-patient it's n_genes x 1)
    - edge_index: COO format edge index
    - edge_weight: edge weights from STRING combined_score
    - y: patient label (optional)

    Args:
        patient_features: DataFrame (patients x genes)
        gene_list: Ordered list of gene names (determines node order)
        string_df: STRING interactions
        labels: Optional target labels

    Returns:
        List of PyG Data objects (one per patient)
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "PyTorch and torch-geometric are required for GCN/GAT models.\n"
            "Install with: pip install torch torch-geometric"
        )

    # Build edge index from STRING
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    edges_src, edges_dst, edge_weights = [], [], []

    for _, row in string_df.iterrows():
        g1, g2, score = row["gene1"], row["gene2"], row["combined_score"]
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            edges_src += [i, j]
            edges_dst += [j, i]  # undirected → both directions
            w = float(score) / 1000.0
            edge_weights += [w, w]

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    # Align patient features to gene_list order
    available_genes = [g for g in gene_list if g in patient_features.columns]
    missing_genes = [g for g in gene_list if g not in patient_features.columns]

    patient_list = []
    for patient_id in patient_features.index:
        # Build feature vector for each node (gene)
        node_features = []
        for gene in gene_list:
            if gene in patient_features.columns:
                val = float(patient_features.loc[patient_id, gene])
            else:
                val = 0.0
            node_features.append([val])

        x = torch.tensor(node_features, dtype=torch.float)  # shape: (n_genes, 1)
        y = None
        if labels is not None and patient_id in labels.index:
            y = torch.tensor([int(labels[patient_id])], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            patient_id=patient_id,
        )
        patient_list.append(data)

    logger.info(
        f"Built {len(patient_list)} patient graphs | "
        f"{len(gene_list)} nodes, {len(edges_src)//2} edges "
        f"({len(missing_genes)} genes without features → zero-filled)"
    )
    return patient_list


def get_graph_stats(G) -> dict:
    """Return summary statistics for a NetworkX graph."""
    try:
        import networkx as nx
    except ImportError:
        return {}

    components = list(nx.connected_components(G))
    degrees = [d for _, d in G.degree()]
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": len(components),
        "largest_component_size": max(len(c) for c in components) if components else 0,
        "isolated_nodes": len(list(nx.isolates(G))),
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "density": float(nx.density(G)),
        "is_connected": nx.is_connected(G),
    }
