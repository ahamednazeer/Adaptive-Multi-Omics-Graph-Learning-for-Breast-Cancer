"""SHAP router — global and per-patient SHAP explanations."""
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models as db_models

router = APIRouter()

@router.get("/global")
def get_global_shap(limit: int = Query(30, ge=1, le=100), db: Session = Depends(get_db)):
    biomarkers = db.query(db_models.Biomarker).order_by(
        db_models.Biomarker.mean_abs_shap.desc()
    ).limit(limit).all()
    if not biomarkers:
        return {"pipeline_ran": False, "features": []}
    return {
        "pipeline_ran": True,
        "features": [
            {
                "name": b.gene, "value": b.mean_shap,
                "mean_abs_shap": b.mean_abs_shap, "type": b.direction,
            }
            for b in biomarkers
        ],
    }

@router.get("/patient/{patient_id}")
def get_patient_shap(patient_id: str, db: Session = Depends(get_db)):
    p = db.query(db_models.PatientPrediction).filter(
        db_models.PatientPrediction.patient_id == patient_id
    ).first()
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    if not p.shap_json:
        raise HTTPException(status_code=404, detail="No SHAP data for this patient")
    return {
        "patient_id": patient_id,
        "prediction": p.prediction_label,
        "probability": p.probability,
        "features": json.loads(p.shap_json),
    }
