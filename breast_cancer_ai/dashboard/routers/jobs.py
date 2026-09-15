"""Jobs, biomarkers, evaluation, validation, SHAP, patients routers."""
from __future__ import annotations
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models as db_models

# ─── Jobs ─────────────────────────────────────────────────
jobs_router = APIRouter()
router = jobs_router  # alias — used by main.py

@jobs_router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(db_models.PipelineJob).filter(db_models.PipelineJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id, "status": job.status,
        "current_step": job.current_step,
        "steps_completed": json.loads(job.steps_completed or "[]"),
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "duration_seconds": job.duration_seconds,
    }

@jobs_router.get("/")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(db_models.PipelineJob).order_by(db_models.PipelineJob.created_at.desc()).limit(10).all()
    return [{"id": j.id, "status": j.status, "created_at": j.created_at.isoformat() if j.created_at else None} for j in jobs]
