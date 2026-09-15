"""Biomarkers router — returns ranked SHAP-based biomarkers from DB."""
import json
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models as db_models

router = APIRouter()

@router.get("/")
def get_biomarkers(
    limit: int = Query(30, ge=1, le=100),
    direction: str = Query("all", description="all|positive|negative"),
    feature_type: str = Query("all", description="all|rna|protein"),
    db: Session = Depends(get_db),
):
    q = db.query(db_models.Biomarker)
    if direction != "all":
        q = q.filter(db_models.Biomarker.direction == direction)
    if feature_type != "all":
        q = q.filter(db_models.Biomarker.feature_type == feature_type)
    biomarkers = q.order_by(db_models.Biomarker.mean_abs_shap.desc()).limit(limit).all()

    if not biomarkers:
        return {"biomarkers": [], "pipeline_ran": False}

    return {
        "biomarkers": [
            {
                "gene": b.gene, "feature_type": b.feature_type,
                "mean_abs_shap": b.mean_abs_shap, "mean_shap": b.mean_shap,
                "direction": b.direction, "pathway": b.pathway, "rank": b.rank,
            }
            for b in biomarkers
        ],
        "pipeline_ran": True,
        "total": len(biomarkers),
    }
