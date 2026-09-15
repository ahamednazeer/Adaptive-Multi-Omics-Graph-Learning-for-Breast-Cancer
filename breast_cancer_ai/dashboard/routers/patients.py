"""Patients router — serves stored patient predictions from SQLite."""
from __future__ import annotations
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models as db_models

router = APIRouter()

@router.get("/")
def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str = Query("", description="Filter by patient_id"),
    db: Session = Depends(get_db),
):
    q = db.query(db_models.PatientPrediction)
    if search:
        q = q.filter(db_models.PatientPrediction.patient_id.contains(search))
    total = q.count()
    patients = q.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "patients": [
            {
                "id": p.id, "patient_id": p.patient_id, "dataset": p.dataset,
                "her2": p.her2, "hr": p.hr, "treatment_arm": p.treatment_arm,
                "prediction": p.prediction, "prediction_label": p.prediction_label,
                "probability": p.probability,
            }
            for p in patients
        ],
    }

@router.get("/{patient_id}")
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    p = db.query(db_models.PatientPrediction).filter(
        db_models.PatientPrediction.patient_id == patient_id
    ).first()
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    shap_features = json.loads(p.shap_json) if p.shap_json else []
    return {
        "id": p.id, "patient_id": p.patient_id, "dataset": p.dataset,
        "her2": p.her2, "hr": p.hr, "treatment_arm": p.treatment_arm,
        "prediction": p.prediction, "prediction_label": p.prediction_label,
        "probability": p.probability, "shap_features": shap_features,
    }

@router.post("/predict")
def predict_patient(request: dict, db: Session = Depends(get_db)):
    """Run prediction for a new patient using the trained model."""
    from pathlib import Path
    from breast_cancer_ai.src.models.predictor import pCRPredictor

    model_dir = "experiments/latest"
    if not Path(f"{model_dir}/predictor_model.joblib").exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run the pipeline first.")

    predictor = pCRPredictor(model_dir)
    predictor.load()

    # Build patient features from request
    features = {}
    features.update(request.get("rna_features") or {})
    features.update(request.get("protein_features") or {})
    features["her2"] = request.get("her2", 0)
    features["hr"] = request.get("hr", 1)

    result = predictor.predict_patient(features)
    patient_id = request.get("patient_id") or f"new_{id(request)}"

    # Store prediction
    db.add(db_models.PatientPrediction(
        patient_id=patient_id, dataset="new",
        her2=request.get("her2"), hr=request.get("hr"),
        treatment_arm=request.get("treatment_arm"),
        prediction=result["prediction"],
        prediction_label=result["prediction_label"],
        probability=result["probability"],
    ))
    db.commit()
    return {**result, "patient_id": patient_id}
