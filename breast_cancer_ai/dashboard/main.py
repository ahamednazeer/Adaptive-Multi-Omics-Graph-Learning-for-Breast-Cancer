"""
FastAPI backend for the Breast Cancer AI platform.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from .database import SessionLocal, engine, Base
from . import models as db_models
from .schemas import (
    PipelineRunRequest, JobResponse, OverviewResponse,
    PredictRequest, PredictResponse, PipelineStatusResponse,
)
from .routers import overview, jobs, pipeline, patients, biomarkers, evaluation, validation, shap

# Create tables
Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Breast Cancer AI API",
    description="Adaptive Multi-Omics Graph Learning for Breast Cancer pCR Prediction",
    version="1.0.0",
)

cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allow_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(overview.router, prefix="/api", tags=["overview"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(patients.router, prefix="/api/patients", tags=["patients"])
app.include_router(biomarkers.router, prefix="/api/biomarkers", tags=["biomarkers"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
app.include_router(validation.router, prefix="/api/validation", tags=["validation"])
app.include_router(shap.router, prefix="/api/shap", tags=["shap"])


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
