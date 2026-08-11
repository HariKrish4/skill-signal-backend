import os
import sys
import logging
import csv
from models import EvaluationData
from service import run_pipeline, compute_overall_score
from transform import transform_evaluation_response
from config import DEVELOPMENT_MODE

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)5s - %(lineno)5d - %(funcName)33s - %(levelname)5s - %(message)s",
)


def print_evaluation_results(
    evaluation: EvaluationData, candidate_name: str = "Candidate"
):
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 80)
    print(f"📊 RESUME EVALUATION RESULTS FOR: {candidate_name}")
    print("=" * 80)

    if not evaluation:
        print("❌ No evaluation data available")
        return

    # Calculate overall score
    total_score = compute_overall_score(evaluation)
    max_score = 0

    if hasattr(evaluation, "scores") and evaluation.scores:
        for category_name, category_data in evaluation.scores.model_dump().items():
            category_score = min(category_data["score"], category_data["max"])
            max_score += category_data["max"]

            # Log warning if score was capped
            if category_score < category_data["score"]:
                print(
                    f"⚠️  Warning: {category_name} score capped from {category_data['score']} to {category_score} (max: {category_data['max']})"
                )

    # Warn if total score was capped at maximum possible value
    max_possible_score = max_score + 20  # 120 (100 categories + 20 bonus)
    if total_score >= max_possible_score:
        print(f"⚠️  Warning: Total score capped at maximum possible value")

    # Overall Score
    print(f"\n🎯 OVERALL SCORE: {total_score:.1f}/{max_score}")

    # Detailed Scores
    print("\n📈 DETAILED SCORES:")
    print("-" * 60)

    if hasattr(evaluation, "scores") and evaluation.scores:
        # Define category maximums
        category_maxes = {
            "open_source": 25,
            "self_projects": 20,
            "production": 25,
            "technical_skills": 10,
            "formatting": 5,
            "impact": 15,
        }

        # Open Source
        if hasattr(evaluation.scores, "open_source") and evaluation.scores.open_source:
            os_score = evaluation.scores.open_source
            capped_score = min(os_score.score, category_maxes["open_source"])
            print(f"🌐 Open Source:          {capped_score}/{os_score.max}")
            print(f"   Evidence: {os_score.evidence}")
            print()

        # Self Projects
        if (
            hasattr(evaluation.scores, "self_projects")
            and evaluation.scores.self_projects
        ):
            sp_score = evaluation.scores.self_projects
            capped_score = min(sp_score.score, category_maxes["self_projects"])
            print(f"🚀 Self Projects:        {capped_score}/{sp_score.max}")
            print(f"   Evidence: {sp_score.evidence}")
            print()

        # Production Experience
        if hasattr(evaluation.scores, "production") and evaluation.scores.production:
            prod_score = evaluation.scores.production
            capped_score = min(prod_score.score, category_maxes["production"])
            print(f"🏢 Production Experience: {capped_score}/{prod_score.max}")
            print(f"   Evidence: {prod_score.evidence}")
            print()

        # Technical Skills
        if (
            hasattr(evaluation.scores, "technical_skills")
            and evaluation.scores.technical_skills
        ):
            tech_score = evaluation.scores.technical_skills
            capped_score = min(tech_score.score, category_maxes["technical_skills"])
            print(f"💻 Technical Skills:     {capped_score}/{tech_score.max}")
            print(f"   Evidence: {tech_score.evidence}")
            print()

        # Formatting
        if hasattr(evaluation.scores, "formatting") and evaluation.scores.formatting:
            fmt_score = evaluation.scores.formatting
            capped_score = min(fmt_score.score, category_maxes["formatting"])
            print(f"📄 Formatting:           {capped_score}/{fmt_score.max}")
            print(f"   Evidence: {fmt_score.evidence}")
            print()

        # Impact
        if hasattr(evaluation.scores, "impact") and evaluation.scores.impact:
            imp_score = evaluation.scores.impact
            capped_score = min(imp_score.score, category_maxes["impact"])
            print(f"⚡ Impact:               {capped_score}/{imp_score.max}")
            print(f"   Evidence: {imp_score.evidence}")
            print()

    # Bonus Points
    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        print(f"\n⭐ BONUS POINTS: {evaluation.bonus_points.total}")
        print("-" * 30)
        print(f"   {evaluation.bonus_points.breakdown}")

    # Deductions
    if (
        hasattr(evaluation, "deductions")
        and evaluation.deductions
        and evaluation.deductions.total > 0
    ):
        print(f"\n⚠️  DEDUCTIONS: -{evaluation.deductions.total}")
        print("-" * 30)
        if evaluation.deductions.reasons:
            print(f"   {evaluation.deductions.reasons}")

    # Key Strengths
    if hasattr(evaluation, "key_strengths") and evaluation.key_strengths:
        print(f"\n✅ KEY STRENGTHS:")
        print("-" * 30)
        for i, strength in enumerate(evaluation.key_strengths, 1):
            print(f"  {i}. {strength}")

    # Areas for Improvement
    if (
        hasattr(evaluation, "areas_for_improvement")
        and evaluation.areas_for_improvement
    ):
        print(f"\n🔧 AREAS FOR IMPROVEMENT:")
        print("-" * 30)
        for i, area in enumerate(evaluation.areas_for_improvement, 1):
            print(f"  {i}. {area}")

    print("\n" + "=" * 80)


def main(pdf_path):
    result = run_pipeline(pdf_path)

    if result is None:
        return None

    score = result["evaluation"]
    resume_data = result["resume_data"]
    github_data = result["github_data"]

    # Print evaluation results in readable format
    print_evaluation_results(score, result["candidate_name"])

    if DEVELOPMENT_MODE:
        csv_row = transform_evaluation_response(
            file_name=os.path.basename(pdf_path),
            evaluation=score,
            resume_data=resume_data,
            github_data=github_data,
        )

        # Write CSV row to file
        csv_path = "resume_evaluations.csv"
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(csv_row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write headers if file doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write the row
            writer.writerow(csv_row)

    return score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score.py <pdf_path>")
        exit(1)
    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' does not exist.")
        exit(1)

    main(pdf_path)
