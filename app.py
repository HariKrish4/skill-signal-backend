"""
REST API for the Hiring Agent resume-to-score pipeline.

Run with:
    uvicorn app:app --reload
"""

import hashlib
import logging
import os
import re
import tempfile
from typing import Dict, Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    "http://localhost:5173,https://skill-signal-frontend-tau.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

BLOB_API_BASE = "https://vercel.com/api/blob"

BLOB_API_HEADERS = {"x-api-version": "12"}

RESUME_STEM_RE = re.compile(r"^[a-f0-9]{64}$")


def _blob_store_id() -> Optional[str]:
    """Extract the Blob store id from the read/write token (matches the SDK)."""
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token:
        return None
    parts = token.split("_")
    return parts[3] if len(parts) > 3 else None


def _upload_to_blob(content: bytes, cache_stem: str) -> str:
    """Upload resume bytes to a private Vercel Blob store and return its URL."""
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token:
        raise HTTPException(
            status_code=500, detail="BLOB_READ_WRITE_TOKEN is not configured"
        )

    response = requests.put(
        f"{BLOB_API_BASE}/",
        params={"pathname": f"resumes/{cache_stem}.pdf"},
        headers={
            "Authorization": f"Bearer {token}",
            "x-vercel-blob-access": "private",
            "x-content-type": "application/pdf",
            "x-allow-overwrite": "1",
            **BLOB_API_HEADERS,
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
    resume_url: Optional[str] = None


@app.get("/")
def root():
    """Service overview and available endpoints."""
    return {
        "service": "Hiring Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "evaluate": "/evaluate",
            "docs": "/docs",
        },
    }


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
        try:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            resume_path = os.path.join(UPLOAD_DIR, f"{cache_stem}.pdf")
            with open(resume_path, "wb") as f:
                f.write(content)
        except OSError as e:
            temp_path = os.path.join(tempfile.gettempdir(), f"{cache_stem}.pdf")
            with open(temp_path, "wb") as f:
                f.write(content)
            logger.warning(
                "Failed to write to %s, falling back to %s: %s",
                UPLOAD_DIR,
                temp_path,
                e,
            )
            resume_path = temp_path

    try:
        result = run_pipeline(resume_path, cache_stem=cache_stem)
    except Exception as e:
        logger.exception("Evaluation pipeline failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

    if result is None:
        raise HTTPException(
            status_code=422, detail="Failed to extract resume data from PDF"
        )

    if _blob_store_id():
        result["resume_url"] = f"/resume/{cache_stem}"

    return result


@app.get("/resume/{cache_stem}")
def get_resume(cache_stem: str):
    """Stream a stored resume PDF back from the private Blob store."""
    if not RESUME_STEM_RE.match(cache_stem):
        raise HTTPException(status_code=404, detail="Stored resume not found")

    store_id = _blob_store_id()
    if not store_id:
        raise HTTPException(status_code=404, detail="Stored resume not found")

    blob_url = (
        f"https://{store_id}.private.blob.vercel-storage.com/resumes/{cache_stem}.pdf"
    )
    upstream = requests.get(
        blob_url,
        headers={"Authorization": f"Bearer {os.environ['BLOB_READ_WRITE_TOKEN']}"},
        timeout=60,
    )
    if upstream.status_code == 404:
        raise HTTPException(status_code=404, detail="Stored resume not found")
    upstream.raise_for_status()

    return StreamingResponse(
        upstream.iter_content(chunk_size=8192),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{cache_stem}.pdf"',
            "Cache-Control": "public, max-age=3600",
        },
    )
