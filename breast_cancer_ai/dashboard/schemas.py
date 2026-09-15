"""Pydantic v2 schemas for all API request/response types."""
from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel
from datetime import datetime


# ─── Pipeline ─────────────────────────────────────────────
class PipelineRunRequest(BaseModel):
    steps: Optional[list[str]] = None  # None = run all 10

class StepStatus(BaseModel):
    step_name: str
    status: str
    message: Optional[str] = None
    duration_seconds: Optional[float] = None

class PipelineStatusResponse(BaseModel):
    job_id: str
    status: str
    current_step: Optional[str] = None
    steps_completed: list[str] = []
    steps_total: int = 10
    error_message: Optional[str] = None

class JobResponse(BaseModel):
    id: str
    status: str
    current_step: Optional[str] = None
    steps_completed: list[str] = []
    result: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    duration_seconds: Optional[float] = None


# ─── Overview ─────────────────────────────────────────────
class DatasetInfo(BaseModel):
    name: str
    n_patients: int
    n_features: int
    role: str
    size_mb: float

class OverviewResponse(BaseModel):
    total_patients: int
    total_rna_features: int
    total_protein_features: int
    pcr_rate: float
    datasets: list[DatasetInfo]
    pipeline_ran: bool
    best_model_auc: Optional[float] = None
    last_run: Optional[str] = None


# ─── Predictions ──────────────────────────────────────────
class PredictRequest(BaseModel):
    patient_id: Optional[str] = None
    her2: int = 0
    hr: int = 1
    treatment_arm: str = "Control"
    # RNA features (optional subset)
    rna_features: Optional[dict[str, float]] = None
    # Protein features (optional)
    protein_features: Optional[dict[str, float]] = None

class PredictResponse(BaseModel):
    patient_id: str
    prediction: int
    prediction_label: str
    probability: float
    model_type: str
    top_features: list[dict] = []


# ─── Patients ─────────────────────────────────────────────
class PatientResponse(BaseModel):
    id: int
    patient_id: str
    dataset: str
    her2: Optional[int] = None
    hr: Optional[int] = None
    treatment_arm: Optional[str] = None
    prediction: Optional[int] = None
    prediction_label: Optional[str] = None
    probability: Optional[float] = None


# ─── Biomarkers ───────────────────────────────────────────
class BiomarkerResponse(BaseModel):
    gene: str
    feature_type: str
    mean_abs_shap: float
    mean_shap: float
    direction: str
    pathway: Optional[str] = None
    rank: int


# ─── Models ───────────────────────────────────────────────
class ModelMetricsResponse(BaseModel):
    model_name: str
    split: str
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    confusion_matrix: Optional[dict] = None
    roc_curve: Optional[list[dict]] = None
