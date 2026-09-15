"""
Graph validation — checks connectivity, coverage, and isolated nodes.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def validate_graph(G, gene_list: list[str]) -> dict:
    """
    Comprehensive validation of the biological interaction graph.

    Checks:
    - Connectivity
    - Isolated nodes
    - Missing gene mappings
    - Edge distribution
    """
    try:
        import networkx as nx
        import numpy as np
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    report = {
        "passed": True,
        "issues": [],
        "warnings": [],
        "stats": {},
    }

    # Basic stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    report["stats"]["n_nodes"] = n_nodes
    report["stats"]["n_edges"] = n_edges

    # Connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        n_cc = len(components)
        largest_cc = max(len(c) for c in components)
        report["warnings"].append(
            f"Graph is disconnected: {n_cc} components. "
            f"Largest component has {largest_cc} nodes ({largest_cc/n_nodes:.1%})."
        )
        report["stats"]["n_components"] = n_cc
        report["stats"]["largest_component_size"] = largest_cc
    else:
        report["stats"]["n_components"] = 1
        report["stats"]["is_connected"] = True

    # Isolated nodes
    isolated = list(nx.isolates(G))
    if isolated:
        report["warnings"].append(
            f"{len(isolated)} isolated nodes (no edges): {isolated[:5]}..."
        )
    report["stats"]["isolated_nodes"] = len(isolated)

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    report["stats"]["avg_degree"] = round(float(np.mean(degrees)), 2) if degrees else 0
    report["stats"]["median_degree"] = float(np.median(degrees)) if degrees else 0
    report["stats"]["max_degree"] = int(max(degrees)) if degrees else 0

    # Missing gene coverage
    genes_in_graph = set(G.nodes())
    missing = [g for g in gene_list if g not in genes_in_graph]
    coverage = 1 - len(missing) / len(gene_list) if gene_list else 0.0
    report["stats"]["gene_coverage"] = round(coverage, 4)
    report["stats"]["missing_genes"] = len(missing)

    if coverage < 0.5:
        report["issues"].append(
            f"Low gene coverage: only {coverage:.1%} of selected genes "
            f"mapped to STRING graph."
        )
        report["passed"] = False
    elif coverage < 0.8:
        report["warnings"].append(
            f"Moderate gene coverage: {coverage:.1%} of genes in graph."
        )

    # Density
    report["stats"]["density"] = round(float(nx.density(G)), 6)

    # Hub nodes (top 10 by degree)
    degree_dict = dict(G.degree())
    top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    report["stats"]["top_hubs"] = [{"gene": g, "degree": d} for g, d in top_hubs]

    if report["passed"] and not report["issues"]:
        logger.info(
            f"Graph validation passed: {n_nodes} nodes, {n_edges} edges, "
            f"coverage={coverage:.1%}"
        )
    else:
        logger.warning(
            f"Graph validation issues: {report['issues']}"
        )

    return report
