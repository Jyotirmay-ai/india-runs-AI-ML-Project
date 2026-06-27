import json
import pandas as pd
from logger import logger


def merge_candidate_data(shortlisted, ranked):
    """Merge full candidate profiles with evaluation results."""
    ranked_by_id = {}
    for r in ranked:
        ranked_by_id[r["candidate_id"]] = r

    merged = []
    for cand in shortlisted:
        cid = cand.get("candidate_id", "")
        eval_data = ranked_by_id.get(cid, {})
        merged.append({"candidate": cand, "evaluation": eval_data})

    logger.info(f"Merged {len(merged)} candidates with evaluation data")
    return merged


def export_ranked_candidates(merged_list, output_dir, formats=("csv", "xlsx")):
    """Export merged candidate+eval data flat to CSV and/or XLSX with HR-friendly columns."""
    rows = []
    for rank, item in enumerate(merged_list, 1):
        cand = item["candidate"]
        eval_data = item["evaluation"]
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        breakdown = eval_data.get("breakdown", {})

        row = {
            # --- Ranking & Core Evaluation ---
            "rank": rank,
            "candidate_id": cand.get("candidate_id", ""),
            "name": profile.get("anonymized_name", ""),
            "total_score": eval_data.get("score", 0),
            "hard_skills_score": breakdown.get("hard_skills", 0),
            "experience_relevance_score": breakdown.get("experience_relevance", 0),
            "behavioral_fit_score": breakdown.get("behavioral_fit", 0),
            "shipper_mindset_score": breakdown.get("shipper_mindset", 0),
            "reasoning": eval_data.get("reasoning", ""),
            "red_flags": eval_data.get("red_flags", ""),

            # --- Profile Basics ---
            "headline": profile.get("headline", ""),
            "current_title": profile.get("current_title", ""),
            "current_company": profile.get("current_company", ""),
            "years_experience": profile.get("years_of_experience", ""),
            "location": profile.get("location", ""),
            "country": profile.get("country", ""),

            # --- Contact / Signals ---
            "email": profile.get("email", ""),
            "phone": profile.get("phone", ""),
            "linkedin": profile.get("linkedin_url", ""),
            "portfolio": profile.get("portfolio_url", ""),
            "github": profile.get("github_url", ""),
            "open_to_work": signals.get("open_to_work_flag", ""),
            "response_rate": signals.get("recruiter_response_rate", ""),
            "profile_completeness": signals.get("profile_completeness_score", ""),
            "expected_salary_lpa": signals.get("expected_salary_range_inr_lpa", {}),
            "preferred_work_mode": signals.get("preferred_work_mode", ""),
            "willing_to_relocate": signals.get("willing_to_relocate", ""),
            "notice_period_days": signals.get("notice_period_days", ""),

            # --- Skills (as JSON string for full detail) ---
            "skills_json": json.dumps(cand.get("skills", []), ensure_ascii=False),

            # --- Career History (as JSON string) ---
            "career_history_json": json.dumps(cand.get("career_history", []), ensure_ascii=False),

            # --- Education (as JSON string) ---
            "education_json": json.dumps(cand.get("education", []), ensure_ascii=False),

            # --- Certifications & Languages (as JSON strings) ---
            "certifications_json": json.dumps(cand.get("certifications", []), ensure_ascii=False),
            "languages_json": json.dumps(cand.get("languages", []), ensure_ascii=False),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns: key info first, then details
    key_cols = [
        "rank", "candidate_id", "name", "total_score",
        "hard_skills_score", "experience_relevance_score",
        "behavioral_fit_score", "shipper_mindset_score",
        "reasoning", "red_flags",
        "headline", "current_title", "current_company",
        "years_experience", "location", "country",
        "email", "phone", "linkedin", "portfolio", "github",
        "open_to_work", "response_rate", "profile_completeness",
        "expected_salary_lpa", "preferred_work_mode",
        "willing_to_relocate", "notice_period_days",
        "skills_json", "career_history_json", "education_json",
        "certifications_json", "languages_json"
    ]
    # Keep any extra columns not in key_cols at the end
    extra_cols = [c for c in df.columns if c not in key_cols]
    df = df[key_cols + extra_cols]

    csv_path = output_dir / "6_ranked_candidates.csv"
    xlsx_path = output_dir / "6_ranked_candidates.xlsx"

    if "csv" in formats:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Exported CSV: {csv_path} ({len(df)} rows)")

    if "xlsx" in formats:
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        logger.info(f"Exported XLSX: {xlsx_path} ({len(df)} rows)")