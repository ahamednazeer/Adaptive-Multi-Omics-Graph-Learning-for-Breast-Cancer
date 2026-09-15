"""Overview router — returns dataset summary and system status."""
import json
from pathlib import Path
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models as db_models
from ..schemas import OverviewResponse, DatasetInfo

router = APIRouter()


@router.get("/overview", response_model=OverviewResponse)
def get_overview(db: Session = Depends(get_db)):
    # Check if pipeline has been run by looking for step result files
    summary_path = Path("experiments/latest/pipeline_summary.json")
    step_inspection = Path("experiments/latest/step_data_inspection.json")

    pipeline_ran = summary_path.exists() or step_inspection.exists()
    best_auc = None
    last_run = None

    if pipeline_ran:
        try:
            best_model = db.query(db_models.ModelResult).order_by(
                db_models.ModelResult.roc_auc.desc()
            ).first()
            if best_model:
                best_auc = best_model.roc_auc

            latest_job = db.query(db_models.PipelineJob).filter(
                db_models.PipelineJob.status == "done"
            ).order_by(db_models.PipelineJob.completed_at.desc()).first()
            if latest_job and latest_job.completed_at:
                last_run = latest_job.completed_at.isoformat()
        except Exception:
            pass

    # Real dataset shapes from the inspection step result
    ispy2_patients, ispy2_rna, ispy2_prot = 736, 19134, 139
    pcr_rate = 0.348

    if step_inspection.exists():
        try:
            with open(step_inspection) as f:
                d = json.load(f)
            ispy2_patients = d.get("ispy2_n_patients", 736)
            ispy2_rna = d.get("ispy2_rna_features", 19134)
            ispy2_prot = d.get("ispy2_protein_features", 139)
            pcr_rate = d.get("ispy2_pcr_rate", 0.348)
        except Exception:
            pass

    datasets = [
        DatasetInfo(name="ISPY2", n_patients=736, n_features=ispy2_rna + ispy2_prot,
                    role="Training / Test", size_mb=98.0),
        DatasetInfo(name="GSE122630", n_patients=34, n_features=19502,
                    role="Temporal Validation", size_mb=10.0),
        DatasetInfo(name="GSE240671", n_patients=61, n_features=58243,
                    role="External Validation", size_mb=18.0),
    ]

    return OverviewResponse(
        total_patients=ispy2_patients,
        total_rna_features=ispy2_rna,
        total_protein_features=ispy2_prot,
        pcr_rate=pcr_rate,
        datasets=datasets,
        pipeline_ran=pipeline_ran,
        best_model_auc=best_auc,
        last_run=last_run,
    )
