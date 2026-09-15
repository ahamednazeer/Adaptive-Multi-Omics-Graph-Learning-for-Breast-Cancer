"""Pipeline router — run and monitor the 10-step ML pipeline."""
from __future__ import annotations
import json
import threading
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from ..database import get_db
from .. import models as db_models
from ..schemas import PipelineRunRequest, PipelineStatusResponse, JobResponse

router = APIRouter()

# Global lock and abort signal for pipeline execution
_pipeline_lock = threading.Lock()
_abort_event = threading.Event()


def _run_pipeline_task(job_id: str, steps: list | None):
    """Background task that runs the full pipeline and stores results."""
    import sys
    import os

    _abort_event.clear()

    # Add project root to path
    project_root = str(__file__)
    for _ in range(4):
        project_root = os.path.dirname(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from breast_cancer_ai.src.pipeline.orchestrator import PipelineOrchestrator
    from breast_cancer_ai.dashboard.database import SessionLocal

    db = SessionLocal()
    try:
        def status_callback(step: str, status: str, message: str):
            record = db.query(db_models.PipelineStep).filter(
                db_models.PipelineStep.job_id == job_id,
                db_models.PipelineStep.step_name == step,
            ).first()
            if record:
                record.status = status
                record.message = message
                if status == "done":
                    record.completed_at = datetime.utcnow()
                db.commit()

            job = db.query(db_models.PipelineJob).filter(
                db_models.PipelineJob.id == job_id
            ).first()
            if job:
                job.current_step = step
                if status == "done":
                    completed = json.loads(job.steps_completed or "[]")
                    if step not in completed:
                        completed.append(step)
                    job.steps_completed = json.dumps(completed)
                db.commit()

        orchestrator = PipelineOrchestrator(
            config_path="breast_cancer_ai/configs/data.yaml",
            model_config_path="breast_cancer_ai/configs/model.yaml",
            output_dir="experiments/latest",
            status_callback=status_callback,
            abort_check=lambda: _abort_event.is_set(),
        )

        job = db.query(db_models.PipelineJob).filter(
            db_models.PipelineJob.id == job_id
        ).first()
        if job:
            job.status = "running"
            db.commit()

        results = orchestrator.run(steps=steps)

        if _abort_event.is_set():
            if job:
                job.status = "stopped"
                job.error_message = "Pipeline stopped by user"
                job.completed_at = datetime.utcnow()
                db.commit()
            return

        # Save final results and model metrics to DB
        _persist_results(db, job_id, results)

        if job:
            job.status = "done"
            job.completed_at = datetime.utcnow()
            job.result_json = json.dumps(results, default=str)
            db.commit()

    except Exception as e:
        db = SessionLocal()
        job = db.query(db_models.PipelineJob).filter(
            db_models.PipelineJob.id == job_id
        ).first()
        if job:
            job.status = "error"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()


def _persist_results(db: Session, job_id: str, results: dict):
    """Store pipeline results into DB tables for fast API queries."""
    from .. import models as dbm

    # Model metrics
    for model_key, step_data in [
        ("splitting_baseline", results.get("splitting_baseline", {})),
        ("adaptive_fusion", results.get("adaptive_fusion", {})),
    ]:
        for model_name, metrics in [
            ("Logistic Regression", step_data.get("baseline_lr", {})),
            ("XGBoost Baseline", step_data.get("baseline_xgb", {})),
            ("Fused Model", step_data.get("fused_model_metrics", {})),
        ]:
            if metrics and "roc_auc" in metrics:
                db.add(dbm.ModelResult(
                    model_name=model_name, split="test",
                    accuracy=metrics.get("accuracy"),
                    f1=metrics.get("f1"),
                    roc_auc=metrics.get("roc_auc"),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    confusion_matrix_json=json.dumps(metrics.get("confusion_matrix", {})),
                    roc_curve_json=json.dumps(metrics.get("roc_curve", [])),
                ))

    # Biomarkers
    expl = results.get("explainability", {})
    all_bm = expl.get("top_positive_biomarkers", []) + expl.get("top_negative_biomarkers", [])
    for rank, bm in enumerate(all_bm, start=1):
        ftype = "protein" if "." in bm["name"] and bm["name"][0].isupper() else "rna"
        db.add(dbm.Biomarker(
            gene=bm["name"], feature_type=ftype,
            mean_abs_shap=bm.get("mean_abs_shap", abs(bm.get("mean_shap", 0))),
            mean_shap=bm.get("mean_shap", 0),
            direction=bm.get("type", "positive"),
            rank=rank,
        ))

    # Validation results
    val = results.get("temporal_external_validation", {})
    if val.get("external_validation"):
        db.add(dbm.ValidationResult(
            validation_type="external", dataset="GSE240671",
            result_json=json.dumps(val["external_validation"], default=str),
        ))
    if val.get("temporal"):
        db.add(dbm.ValidationResult(
            validation_type="temporal", dataset="GSE122630",
            result_json=json.dumps(val["temporal"], default=str),
        ))

    db.commit()


@router.post("/run", response_model=JobResponse)
def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Start the full pipeline as a background job."""
    job_id = str(uuid.uuid4())

    STEP_NAMES = [
        "data_inspection","preprocessing","splitting_baseline","feature_selection",
        "graph_construction","gcn_gat_training","adaptive_fusion",
        "explainability","temporal_external_validation","evaluation",
    ]
    step_list = request.steps or STEP_NAMES

    job = db_models.PipelineJob(
        id=job_id, status="pending", steps_total=len(step_list),
        steps_completed="[]",
    )
    db.add(job)

    for step in step_list:
        db.add(db_models.PipelineStep(
            job_id=job_id, step_name=step, status="pending"
        ))
    db.commit()

    background_tasks.add_task(_run_pipeline_task, job_id, request.steps)
    return JobResponse(id=job_id, status="pending", steps_completed=[])


@router.get("/status", response_model=PipelineStatusResponse)
def get_pipeline_status(db: Session = Depends(get_db)):
    """Get the most recent pipeline job status."""
    job = db.query(db_models.PipelineJob).order_by(
        db_models.PipelineJob.created_at.desc()
    ).first()
    if not job:
        return PipelineStatusResponse(job_id="none", status="not_started")
    return PipelineStatusResponse(
        job_id=job.id, status=job.status,
        current_step=job.current_step,
        steps_completed=json.loads(job.steps_completed or "[]"),
        error_message=job.error_message,
    )


@router.get("/steps")
def get_pipeline_steps(db: Session = Depends(get_db)):
    """Get all step statuses for the latest job."""
    job = db.query(db_models.PipelineJob).order_by(
        db_models.PipelineJob.created_at.desc()
    ).first()
    if not job:
        return {"steps": []}
    steps = db.query(db_models.PipelineStep).filter(
        db_models.PipelineStep.job_id == job.id
    ).all()
    return {"steps": [
        {
            "step_name": s.step_name, "status": s.status,
            "message": s.message, "duration_seconds": s.duration_seconds,
        }
        for s in steps
    ]}


@router.post("/stop")
def stop_pipeline(db: Session = Depends(get_db)):
    """Stop/cancel the currently running pipeline immediately."""
    global _abort_event
    _abort_event.set()

    # Find active or pending jobs and mark them stopped
    jobs = db.query(db_models.PipelineJob).filter(
        db_models.PipelineJob.status.in_(["running", "pending"])
    ).all()

    for job in jobs:
        job.status = "stopped"
        job.error_message = "Pipeline stopped by user"
        job.completed_at = datetime.utcnow()
        # Mark running or pending steps as stopped
        steps = db.query(db_models.PipelineStep).filter(
            db_models.PipelineStep.job_id == job.id,
            db_models.PipelineStep.status.in_(["running", "pending"])
        ).all()
        for s in steps:
            s.status = "stopped"
            s.message = "Cancelled by user"

    db.commit()
    return {"status": "stopped", "message": "Pipeline cancelled successfully"}

