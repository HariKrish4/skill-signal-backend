"""
REST API for the Hiring Agent resume-to-score pipeline.

Run with:
    uvicorn app:app --reload
"""

import hashlib
import logging
import os
from typing import Dict, Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import DEVELOPMENT_MODE
from models import EvaluationData, JSONResume
from prompt import DEFAULT_MODEL, PROVIDER
from service import run_pipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hiring Agent API",
    description="Resume-to-score pipeline: upload a resume PDF and get an explainable evaluation.",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

BLOB_API_BASE = "https://api.vercel.com/v1/blob"


def _upload_to_blob(content: bytes, cache_stem: str) -> str:
    """Upload resume bytes to a private Vercel Blob store and return its URL."""
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token:
        raise HTTPException(
            status_code=500, detail="BLOB_READ_WRITE_TOKEN is not configured"
        )

    response = requests.put(
        f"{BLOB_API_BASE}/upload",
        params={"pathname": f"resumes/{cache_stem}.pdf"},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/pdf",
        },
        data=content,
        timeout=120,
    )
    if response.status_code != 200:
        logger.error("Vercel Blob upload failed: %s", response.text)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to store resume: {response.text[:200]}",
        )
    return response.json()["url"]


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    development_mode: bool


class EvaluationResponse(BaseModel):
    candidate_name: str
    overall_score: float
    evaluation: EvaluationData
    resume_data: Optional[JSONResume] = None
    github_data: Optional[Dict] = None
    cache_used: bool


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check returning the configured LLM provider and model."""
    return HealthResponse(
        status="ok",
        provider=PROVIDER,
        model=DEFAULT_MODEL,
        development_mode=DEVELOPMENT_MODE,
    )


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(file: UploadFile = File(...)):
    """Evaluate a resume PDF and return the structured evaluation result."""
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    cache_stem = hashlib.sha256(content).hexdigest()

    if os.environ.get("BLOB_READ_WRITE_TOKEN"):
        resume_path = _upload_to_blob(content, cache_stem)
    else:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        resume_path = os.path.join(UPLOAD_DIR, f"{cache_stem}.pdf")
        with open(resume_path, "wb") as f:
            f.write(content)

    try:
        result = run_pipeline(resume_path, cache_stem=cache_stem)
    except Exception as e:
        logger.exception("Evaluation pipeline failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

    if result is None:
        raise HTTPException(
            status_code=422, detail="Failed to extract resume data from PDF"
        )

    return result
