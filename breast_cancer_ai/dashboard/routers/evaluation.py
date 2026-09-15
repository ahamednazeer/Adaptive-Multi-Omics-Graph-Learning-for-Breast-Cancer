"""Evaluation router — model metrics, ROC curves, confusion matrices."""
import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models as db_models

router = APIRouter()

@router.get("/comparison")
def get_model_comparison(db: Session = Depends(get_db)):
    results = db.query(db_models.ModelResult).filter(
        db_models.ModelResult.split == "test"
    ).all()
    if not results:
        return {"models": [], "pipeline_ran": False}
    return {
        "models": [
            {
                "model_name": r.model_name, "split": r.split,
                "accuracy": r.accuracy, "f1": r.f1, "roc_auc": r.roc_auc,
                "precision": r.precision, "recall": r.recall,
            }
            for r in results
        ],
        "pipeline_ran": True,
    }

@router.get("/{model_name}")
def get_model_detail(model_name: str, db: Session = Depends(get_db)):
    r = db.query(db_models.ModelResult).filter(
        db_models.ModelResult.model_name == model_name,
        db_models.ModelResult.split == "test",
    ).first()
    if not r:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Model not found or pipeline not run")
    return {
        "model_name": r.model_name, "split": r.split,
        "accuracy": r.accuracy, "f1": r.f1, "roc_auc": r.roc_auc,
        "precision": r.precision, "recall": r.recall,
        "confusion_matrix": json.loads(r.confusion_matrix_json) if r.confusion_matrix_json else None,
        "roc_curve": json.loads(r.roc_curve_json) if r.roc_curve_json else [],
    }

@router.get("/")
def get_evaluation_summary(db: Session = Depends(get_db)):
    """Return full evaluation including CV results from step JSON."""
    import json
    from pathlib import Path
    step_path = Path("experiments/latest/step_evaluation.json")
    cv_data = {}
    if step_path.exists():
        with open(step_path) as f:
            cv_data = json.load(f)
    results = db.query(db_models.ModelResult).all()
    return {
        "model_results": [
            {"model_name": r.model_name, "split": r.split, "accuracy": r.accuracy,
             "f1": r.f1, "roc_auc": r.roc_auc, "precision": r.precision, "recall": r.recall}
            for r in results
        ],
        "cross_validation": cv_data.get("cross_validation", {}),
        "pipeline_ran": len(results) > 0,
    }
