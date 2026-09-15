"""SQLAlchemy ORM models — stores pipeline results in SQLite."""
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.sql import func
from .database import Base


class PipelineJob(Base):
    __tablename__ = "pipeline_jobs"
    id = Column(String, primary_key=True)
    status = Column(String, default="pending")       # pending|running|done|error
    current_step = Column(String, nullable=True)
    steps_completed = Column(Text, default="[]")     # JSON list of step names
    steps_total = Column(Integer, default=10)
    result_json = Column(Text, nullable=True)        # Full results JSON
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)


class PipelineStep(Base):
    __tablename__ = "pipeline_steps"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, index=True)
    step_name = Column(String)
    status = Column(String, default="pending")       # pending|running|done|error
    message = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class PatientPrediction(Base):
    __tablename__ = "patient_predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, index=True)
    dataset = Column(String, default="ispy2")
    her2 = Column(Integer, nullable=True)
    hr = Column(Integer, nullable=True)
    treatment_arm = Column(String, nullable=True)
    prediction = Column(Integer, nullable=True)      # 0 or 1
    prediction_label = Column(String, nullable=True) # "pCR" or "No pCR"
    probability = Column(Float, nullable=True)
    shap_json = Column(Text, nullable=True)          # Per-patient SHAP values
    created_at = Column(DateTime, server_default=func.now())


class ModelResult(Base):
    __tablename__ = "model_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, index=True)
    split = Column(String, default="test")           # train|val|test|cv
    accuracy = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    confusion_matrix_json = Column(Text, nullable=True)
    roc_curve_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class Biomarker(Base):
    __tablename__ = "biomarkers"
    id = Column(Integer, primary_key=True, autoincrement=True)
    gene = Column(String, index=True)
    feature_type = Column(String)   # rna|protein|clinical|treatment
    mean_abs_shap = Column(Float)
    mean_shap = Column(Float)
    direction = Column(String)      # positive|negative
    pathway = Column(String, nullable=True)
    rank = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())


class ValidationResult(Base):
    __tablename__ = "validation_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    validation_type = Column(String)  # external|temporal|resistance
    dataset = Column(String)
    result_json = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
