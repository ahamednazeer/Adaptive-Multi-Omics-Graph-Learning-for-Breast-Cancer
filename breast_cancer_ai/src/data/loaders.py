"""
Data loaders for all three breast cancer datasets.
Based on real column audit — uses exact column names from each CSV.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ─── Real column schemas (from dataset audit) ────────────────────────────────

ISPY2_RNA_PREFIX = "RNA__"
ISPY2_PROT_PREFIX = "PROT__"
ISPY2_CLINICAL_COLS = ["her2", "hr"]
ISPY2_TREATMENT_COL = "arm"
ISPY2_PATIENT_ID = "patient_id"
ISPY2_TARGET = "pcr"

GSE122630_META_COLS = [
    "Sample_ID", "Patient_ID", "Timepoint", "Timepoint_Order",
    "Timepoint_Description", "Number_of_Timepoints",
]

GSE240671_META_COLS = [
    "GSM_ID", "Patient_Number", "Patient_ID", "Library_ID", "Timing",
    "RCB_Category", "pCR_Status", "pCR_Binary", "Molecular_Category",
    "NAC_Category", "NAC_Herceptin", "Age_Diagnostic", "Menopausal_Status",
    "Grade", "Node_Invasion", "Sequencing_Batch",
]


def _cfg_path(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        candidates = [
            Path("breast_cancer_ai") / config_path,
            Path(__file__).resolve().parent.parent.parent / config_path,
            Path(__file__).resolve().parent.parent.parent / "configs" / "data.yaml",
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
    with open(p) as f:
        return yaml.safe_load(f)


def _resolve_csv_path(raw_path: str) -> str:
    p = Path(raw_path)
    if p.exists():
        return str(p)
    p_name = p.name
    candidates = [
        Path(p_name),
        Path("..") / p_name,
        Path("../..") / p_name,
        Path(__file__).resolve().parent.parent.parent.parent / p_name,
        Path(__file__).resolve().parent.parent.parent / p_name,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return raw_path


# ─── ISPY2 ───────────────────────────────────────────────────────────────────

def load_ispy2(
    path: Optional[str] = None,
    config_path: str = "configs/data.yaml",
) -> dict[str, pd.DataFrame]:
    """
    Load ISPY2 multi-omics CSV and split into modality DataFrames.

    Returns dict with keys:
      'rna'       — (736, 19134) log-normalized RNA expression, index=patient_id
      'protein'   — (736, 139)   RPPA protein features, ~20% missing
      'clinical'  — (736, 2)     her2, hr (binary)
      'treatment' — (736, 1)     arm string column (raw, for encoding)
      'target'    — (736,)       pcr label Series (0/1)
      'raw'       — full DataFrame for reference
    """
    cfg = _cfg_path(config_path)
    csv_path = _resolve_csv_path(path or cfg["datasets"]["ispy2"]["path"])
    logger.info(f"Loading ISPY2 from: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df = df.set_index(ISPY2_PATIENT_ID)
    logger.info(f"ISPY2 shape: {df.shape}")

    # Column groups by prefix
    rna_cols = [c for c in df.columns if c.startswith(ISPY2_RNA_PREFIX)]
    prot_cols = [c for c in df.columns if c.startswith(ISPY2_PROT_PREFIX)]

    result = {
        "rna":       df[rna_cols].copy(),
        "protein":   df[prot_cols].copy(),
        "clinical":  df[[c for c in ISPY2_CLINICAL_COLS if c in df.columns]].copy(),
        "treatment": df[[ISPY2_TREATMENT_COL]].copy() if ISPY2_TREATMENT_COL in df.columns else pd.DataFrame(index=df.index),
        "target":    df[ISPY2_TARGET].copy() if ISPY2_TARGET in df.columns else pd.Series(dtype=int, name=ISPY2_TARGET),
        "raw":       df,
    }

    # Strip prefixes from column names for cleaner gene/protein names
    result["rna"].columns = [c.replace(ISPY2_RNA_PREFIX, "") for c in result["rna"].columns]
    result["protein"].columns = [c.replace(ISPY2_PROT_PREFIX, "") for c in result["protein"].columns]

    for key, sub in result.items():
        if hasattr(sub, "shape"):
            logger.info(f"  ISPY2[{key}]: {sub.shape}")

    logger.info(f"  pCR=1: {int(result['target'].sum())}, pCR=0: {int((result['target']==0).sum())}")
    return result


def get_rna_gene_names(ispy2_data: dict) -> list[str]:
    """Return list of RNA gene names (without prefix) from ISPY2."""
    return list(ispy2_data["rna"].columns)


def get_protein_names(ispy2_data: dict) -> list[str]:
    """Return list of protein names (without prefix) from ISPY2."""
    return list(ispy2_data["protein"].columns)


# ─── GSE122630 ───────────────────────────────────────────────────────────────

def load_gse122630(
    path: Optional[str] = None,
    config_path: str = "configs/data.yaml",
) -> dict[str, pd.DataFrame]:
    """
    Load GSE122630 longitudinal RNA dataset (95 samples, 34 patients, T1-T4).
    
    IMPORTANT: Gene columns use Ensembl IDs (ENSG...), NOT gene symbols.
    Use map_ensembl_to_symbol() to convert before comparing with ISPY2.

    Returns dict with:
      'meta'        — (95, 6) metadata (Sample_ID, Patient_ID, Timepoint, ...)
      'expression'  — (95, 19502) Ensembl gene expression
      'by_timepoint'— dict T1/T2/T3/T4 → subset DataFrames
    """
    cfg = _cfg_path(config_path)
    csv_path = _resolve_csv_path(path or cfg["datasets"]["gse122630"]["path"])
    logger.info(f"Loading GSE122630 from: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"GSE122630 shape: {df.shape}")

    meta_cols = [c for c in GSE122630_META_COLS if c in df.columns]
    gene_cols = [c for c in df.columns if c not in meta_cols]

    meta = df[meta_cols].copy()
    expression = df[gene_cols].copy()
    expression.index = list(df["Sample_ID"])

    # Split by timepoint
    by_tp = {}
    for tp in ["T1", "T2", "T3", "T4"]:
        mask = df["Timepoint"] == tp
        sub = df[mask].reset_index(drop=True)
        expr_sub = sub[gene_cols].copy()
        expr_sub.index = list(sub["Sample_ID"])
        by_tp[tp] = {
            "meta": sub[meta_cols].reset_index(drop=True),
            "expression": expr_sub,
        }
        logger.info(f"  GSE122630[{tp}]: {mask.sum()} samples")

    return {
        "meta": meta,
        "expression": expression,
        "by_timepoint": by_tp,
        "gene_cols": gene_cols,
        "meta_cols": meta_cols,
    }


def map_ensembl_to_symbol(
    ensembl_ids: list[str],
    mapping_file: Optional[str] = None,
) -> dict[str, str]:
    """
    Map Ensembl gene IDs to gene symbols.
    
    Args:
        ensembl_ids: List of ENSG... IDs
        mapping_file: Optional path to pre-downloaded mapping TSV
                      (Ensembl_ID, gene_symbol columns)
    
    Returns:
        Dict: ensembl_id → gene_symbol
    """
    if mapping_file and Path(mapping_file).exists():
        mapping_df = pd.read_csv(mapping_file, sep="\t")
        # Expect columns: ensembl_id, gene_symbol (or similar)
        id_col = [c for c in mapping_df.columns if "ensembl" in c.lower() or "ensg" in c.lower()][0]
        sym_col = [c for c in mapping_df.columns if "symbol" in c.lower() or "name" in c.lower()][0]
        return dict(zip(mapping_df[id_col], mapping_df[sym_col]))

    # Try mygene (if installed)
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            ensembl_ids[:len(ensembl_ids)],
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            returnall=False,
        )
        mapping = {}
        for r in results:
            if "symbol" in r:
                mapping[r.get("query", "")] = r["symbol"]
        logger.info(f"Mapped {len(mapping)}/{len(ensembl_ids)} Ensembl IDs via mygene")
        return mapping
    except Exception as e:
        logger.warning(f"mygene mapping failed: {e}. Using Ensembl IDs as-is.")
        return {e: e for e in ensembl_ids}


# ─── GSE240671 ───────────────────────────────────────────────────────────────

def load_gse240671(
    path: Optional[str] = None,
    config_path: str = "configs/data.yaml",
) -> dict[str, pd.DataFrame]:
    """
    Load GSE240671 pre/post treatment RNA dataset.
    122 samples = 61 pre-treatment + 61 post-treatment residual (matched pairs).
    Gene columns use gene symbols (not ENSG IDs).

    Returns dict with:
      'meta'       — (122, 16) clinical/annotation metadata
      'expression' — (122, ~58243) raw count expression
      'pre'        — subset: Pre-treatment samples
      'post'       — subset: Post-treatment residual samples  
      'paired'     — dict patient_id → {pre: row, post: row}
    """
    cfg = _cfg_path(config_path)
    csv_path = _resolve_csv_path(path or cfg["datasets"]["gse240671"]["path"])
    logger.info(f"Loading GSE240671 from: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"GSE240671 shape: {df.shape}")

    meta_cols = [c for c in GSE240671_META_COLS if c in df.columns]
    gene_cols = [c for c in df.columns if c not in meta_cols]

    meta = df[meta_cols]
    if "GSM_ID" in df.columns:
        df.index = list(df["GSM_ID"])
    expression = df[gene_cols]

    # Split by timing
    pre_mask = (df["Timing"] == "Pre-treatment").values
    post_mask = (df["Timing"] == "Post-treatment residual").values
    pre_df = df.iloc[pre_mask]
    post_df = df.iloc[post_mask]

    logger.info(f"  Pre-treatment samples: {pre_mask.sum()}")
    logger.info(f"  Post-treatment samples: {post_mask.sum()}")

    # Build paired dict using only meta columns (instant: 16 cols instead of 58,259)
    paired = {}
    pre_meta = pre_df[meta_cols]
    post_meta = post_df[meta_cols]
    if "Patient_Number" in df.columns:
        for pid in df["Patient_Number"].dropna().unique():
            pre_rows = pre_meta[pre_meta["Patient_Number"] == pid]
            post_rows = post_meta[post_meta["Patient_Number"] == pid]
            if len(pre_rows) > 0 and len(post_rows) > 0:
                paired[pid] = {
                    "pre": pre_rows.iloc[0],
                    "post": post_rows.iloc[0],
                    "pcr": int(pre_rows["pCR_Binary"].iloc[0]) if "pCR_Binary" in pre_rows.columns else None,
                }

    logger.info(f"  Matched pairs: {len(paired)}")

    pre_expr = expression.iloc[pre_mask]
    post_expr = expression.iloc[post_mask]

    return {
        "meta": meta,
        "expression": expression,
        "pre": {
            "meta": pre_df[meta_cols],
            "expression": pre_expr,
        },
        "post": {
            "meta": post_df[meta_cols],
            "expression": post_expr,
        },
        "paired": paired,
        "gene_cols": gene_cols,
        "meta_cols": meta_cols,
    }


# ─── Summary ─────────────────────────────────────────────────────────────────

def get_dataset_summary(config_path: str = "configs/data.yaml") -> dict:
    """Return file-level summary for all datasets without full loading."""
    cfg = _cfg_path(config_path)
    summary = {}
    dataset_files = {
        "ispy2": cfg["datasets"]["ispy2"]["path"],
        "gse122630": cfg["datasets"]["gse122630"]["path"],
        "gse240671": cfg["datasets"]["gse240671"]["path"],
    }
    for name, rel_path in dataset_files.items():
        p = Path(rel_path)
        if p.exists():
            try:
                header = pd.read_csv(p, nrows=0)
                n_cols = len(header.columns)
                with open(p) as f:
                    n_rows = sum(1 for _ in f) - 1
                summary[name] = {
                    "exists": True,
                    "path": str(p),
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
                }
            except Exception as ex:
                summary[name] = {"exists": True, "path": str(p), "error": str(ex)}
        else:
            summary[name] = {"exists": False, "path": str(p)}
    return summary
