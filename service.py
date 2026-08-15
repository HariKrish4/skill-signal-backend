"""
Reusable resume-evaluation pipeline service.

Extracted from score.py so both the CLI (score.py) and the REST API
(app.py) can run the same end-to-end flow and get structured results.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from pdf import PDFHandler
from github import fetch_and_display_github_info
from models import JSONResume, EvaluationData
from evaluator import ResumeEvaluator
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from transform import (
    convert_json_resume_to_text,
    convert_github_data_to_text,
    convert_blog_data_to_text,
)
from config import DEVELOPMENT_MODE

logger = logging.getLogger(__name__)

CACHE_DIR = "cache"

BLOB_HOST = "blob.vercel-storage.com"


def _materialize_blob(pdf_path: str, cache_stem: str) -> str:
    """Download a Vercel Blob object to a local temp file and return its path.

    Local file paths are returned unchanged so the CLI keeps working. Private
    blobs are fetched directly with the read/write token and written to the
    writable temp directory.
    """
    if BLOB_HOST not in pdf_path:
        return pdf_path

    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token:
        raise ValueError(
            "BLOB_READ_WRITE_TOKEN is required to download from Vercel Blob"
        )

    content_resp = requests.get(
        pdf_path,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    content_resp.raise_for_status()

    temp_path = os.path.join(tempfile.gettempdir(), f"{cache_stem}.pdf")
    with open(temp_path, "wb") as f:
        f.write(content_resp.content)
    return temp_path


def is_valid_resume_data(resume_data: JSONResume) -> bool:
    """Check if the resume data has at least some extracted core content."""
    if not resume_data:
        return False
    core_sections = [
        resume_data.basics,
        resume_data.work,
        resume_data.education,
        resume_data.skills,
        resume_data.projects,
    ]
    return any(section is not None for section in core_sections)


def find_profile(profiles, network):
    if not profiles:
        return None
    return next(
        (p for p in profiles if p.network and p.network.lower() == network.lower()),
        None,
    )


def _evaluate_resume(
    resume_data: JSONResume, github_data: dict = None, blog_data: dict = None
) -> Optional[EvaluationData]:
    """Evaluate the resume using AI and return the evaluation data."""
    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, model_params=model_params)

    resume_text = convert_json_resume_to_text(resume_data)

    if github_data:
        github_text = convert_github_data_to_text(github_data)
        resume_text += github_text

    if blog_data:
        blog_text = convert_blog_data_to_text(blog_data)
        resume_text += blog_text

    return evaluator.evaluate_resume(resume_text)


def compute_overall_score(evaluation: Optional[EvaluationData]) -> float:
    """Compute the overall score from an evaluation (categories + bonus - deductions)."""
    if not evaluation:
        return 0.0

    total_score = 0
    max_score = 0

    if evaluation.scores:
        for category_data in evaluation.scores.model_dump().values():
            category_score = min(category_data["score"], category_data["max"])
            total_score += category_score
            max_score += category_data["max"]

    if evaluation.bonus_points:
        total_score += evaluation.bonus_points.total

    if evaluation.deductions:
        total_score -= evaluation.deductions.total

    max_possible_score = max_score + 20  # 120 (100 categories + 20 bonus)
    if total_score > max_possible_score:
        total_score = max_possible_score

    return total_score


def _load_cached_resume(cache_filename: str) -> Optional[JSONResume]:
    """Load a cached JSONResume if it exists and is valid."""
    if not (DEVELOPMENT_MODE and os.path.exists(cache_filename)):
        return None

    print(f"Loading cached data from {cache_filename}")
    try:
        cached_data = json.loads(Path(cache_filename).read_text(encoding="utf-8"))
        loaded_resume = JSONResume(**cached_data)
        if not is_valid_resume_data(loaded_resume):
            raise ValueError("Cached resume data contains no core content")
        return loaded_resume
    except Exception as e:
        print(f"⚠️ Warning: Invalid cache file {cache_filename}: {e}")
        print("Ignoring cache and reprocessing PDF...")
        try:
            os.remove(cache_filename)
        except Exception as delete_err:
            print(f"Failed to delete invalid cache file {cache_filename}: {delete_err}")
    return None


def _save_cached_resume(cache_filename: str, resume_data: JSONResume) -> None:
    if not is_valid_resume_data(resume_data):
        logger.warning(
            "Newly extracted resume data is empty/invalid. Skipping cache write."
        )
        return
    try:
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        Path(cache_filename).write_text(
            json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Failed to cache resume data to %s: %s", cache_filename, e)


def _load_cached_github(github_cache_filename: str) -> Optional[dict]:
    """Load cached GitHub data if it exists and is valid."""
    if not (DEVELOPMENT_MODE and os.path.exists(github_cache_filename)):
        return None

    print(f"Loading cached data from {github_cache_filename}")
    try:
        loaded_github = json.loads(
            Path(github_cache_filename).read_text(encoding="utf-8")
        )
        if (
            not isinstance(loaded_github, dict)
            or not loaded_github
            or "profile" not in loaded_github
        ):
            raise ValueError("Cached GitHub data is invalid or empty")
        return loaded_github
    except Exception as e:
        print(f"⚠️ Warning: Invalid GitHub cache file {github_cache_filename}: {e}")
        print("Ignoring GitHub cache and refetching...")
        try:
            os.remove(github_cache_filename)
        except Exception as delete_err:
            print(
                f"Failed to delete invalid GitHub cache file {github_cache_filename}: {delete_err}"
            )
    return None


def _save_cached_github(github_cache_filename: str, github_data: dict) -> None:
    if (
        not github_data
        or not isinstance(github_data, dict)
        or "profile" not in github_data
    ):
        return
    try:
        os.makedirs(os.path.dirname(github_cache_filename), exist_ok=True)
        Path(github_cache_filename).write_text(
            json.dumps(github_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning(
            "Failed to cache GitHub data to %s: %s", github_cache_filename, e
        )


def run_pipeline(
    pdf_path: str, cache_stem: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Run the full resume evaluation pipeline.

    Args:
        pdf_path: Path to the resume PDF.
        cache_stem: Unique stem used for cache filenames. Defaults to the
            PDF basename (matching the CLI behaviour).

    Returns:
        A dict with candidate_name, resume_data, github_data, evaluation,
        overall_score and cache_used, or None if PDF extraction failed.
    """
    if cache_stem is None:
        cache_stem = os.path.basename(pdf_path).replace(".pdf", "")

    pdf_path = _materialize_blob(pdf_path, cache_stem)

    cache_filename = os.path.join(CACHE_DIR, f"resumecache_{cache_stem}.json")
    github_cache_filename = os.path.join(CACHE_DIR, f"githubcache_{cache_stem}.json")

    resume_data = None
    resume_cache_used = False

    cached_resume = _load_cached_resume(cache_filename)
    if cached_resume is not None:
        resume_data = cached_resume
        resume_cache_used = True

    if not resume_cache_used:
        logger.debug(
            f"Extracting data from PDF"
            + (" and caching to " + cache_filename if DEVELOPMENT_MODE else "")
        )
        pdf_handler = PDFHandler()
        resume_data = pdf_handler.extract_json_from_pdf(pdf_path)

        if resume_data is None:
            return None

        if DEVELOPMENT_MODE:
            _save_cached_resume(cache_filename, resume_data)

    github_data = {}
    github_cache_used = False

    cached_github = _load_cached_github(github_cache_filename)
    if cached_github is not None:
        github_data = cached_github
        github_cache_used = True

    if not github_cache_used:
        profiles = []
        if resume_data and hasattr(resume_data, "basics") and resume_data.basics:
            profiles = resume_data.basics.profiles or []
        github_profile = find_profile(profiles, "Github")

        if github_profile:
            print(
                f"Fetching GitHub data"
                + (
                    " and caching to " + github_cache_filename
                    if DEVELOPMENT_MODE
                    else ""
                )
            )
            github_data = fetch_and_display_github_info(github_profile.url)

            if DEVELOPMENT_MODE:
                _save_cached_github(github_cache_filename, github_data)

    evaluation = _evaluate_resume(resume_data, github_data)

    candidate_name = os.path.basename(pdf_path).replace(".pdf", "")
    if (
        resume_data
        and hasattr(resume_data, "basics")
        and resume_data.basics
        and resume_data.basics.name
    ):
        candidate_name = resume_data.basics.name

    return {
        "candidate_name": candidate_name,
        "resume_data": resume_data,
        "github_data": github_data,
        "evaluation": evaluation,
        "overall_score": compute_overall_score(evaluation),
        "cache_used": resume_cache_used and github_cache_used,
    }
