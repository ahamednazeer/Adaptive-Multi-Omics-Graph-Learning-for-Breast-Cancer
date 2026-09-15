"""Validation router — temporal, external, and resistance results."""
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models as db_models

router = APIRouter()

def _load_step(name: str) -> dict:
    p = Path(f"experiments/latest/step_{name}.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

@router.get("/temporal")
def get_temporal_validation(db: Session = Depends(get_db)):
    rec = db.query(db_models.ValidationResult).filter(
        db_models.ValidationResult.validation_type == "temporal"
    ).order_by(db_models.ValidationResult.created_at.desc()).first()
    if not rec:
        return {"pipeline_ran": False, "data": {}}
    return {"pipeline_ran": True, "data": json.loads(rec.result_json)}

@router.get("/external")
def get_external_validation(db: Session = Depends(get_db)):
    rec = db.query(db_models.ValidationResult).filter(
        db_models.ValidationResult.validation_type == "external"
    ).order_by(db_models.ValidationResult.created_at.desc()).first()
    if not rec:
        return {"pipeline_ran": False, "data": {}}
    return {"pipeline_ran": True, "data": json.loads(rec.result_json)}

@router.get("/resistance")
def get_resistance_analysis(db: Session = Depends(get_db)):
    step = _load_step("temporal_external_validation")
    resistance = step.get("resistance_analysis", {})
    if not resistance:
        return {"pipeline_ran": False, "data": {}}
    return {"pipeline_ran": True, "data": resistance}

@router.get("/")
def get_all_validation(db: Session = Depends(get_db)):
    step = _load_step("temporal_external_validation")
    return {
        "pipeline_ran": bool(step),
        "temporal": step.get("temporal", {}),
        "external": step.get("external_validation", {}),
        "resistance": step.get("resistance_analysis", {}),
    }
