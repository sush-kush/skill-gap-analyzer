"""
src/scoring/score_engine.py

Stage 8 — Scoring Engine
--------------------------
Computes three meaningful, mathematically-defined metrics from a MatchResult:

  1. Skill Match Percentage  — simple ratio of matched skills
  2. Readiness Score         — weighted score accounting for skill importance
  3. Skill Gap Score         — complement of readiness score (the "gap")

Also produces:
  - Readiness label (Beginner / Developing / Intermediate / Advanced / Job-Ready)
  - Per-category readiness breakdown
  - NLP-based similarity score from the TF-IDF module

─────────────────────────────────────────────────────────────
MATHEMATICAL DEFINITIONS
─────────────────────────────────────────────────────────────

Skill Match Percentage (SMP):
    SMP = (n_matched / n_total_required) × 100

Effective Match Numerator (for weighted score):
    Each matched skill   → contributes its full importance weight
    Each partial skill   → contributes 0.5 × its importance weight
    Each missing skill   → contributes 0

Readiness Score (RS):
    RS = [Σ weight(matched_i) + Σ 0.5 × weight(partial_j)]
         ─────────────────────────────────────────────────── × 100
                  Σ weight(all_required_k)

Skill Gap Score (SGS):
    SGS = 100 − RS

Interpretation:
    RS 0–29   → Beginner
    RS 30–49  → Developing
    RS 50–69  → Intermediate
    RS 70–84  → Advanced
    RS 85–100 → Job-Ready
─────────────────────────────────────────────────────────────
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from matching.skill_matcher import MatchResult, SkillMatch

PARTIAL_WEIGHT_FACTOR = 0.5    # partial match = 50 % of full credit


# ── Score result dataclass ────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """
    Holds all computed score metrics for one user-role comparison.
    """
    target_role:          str
    total_required:       int
    n_matched:            int
    n_partial:            int
    n_missing:            int

    skill_match_pct:      float    # % of required skills matched (binary)
    readiness_score:      float    # weighted readiness 0–100
    gap_score:            float    # 100 − readiness_score
    readiness_label:      str      # Beginner / Developing / Intermediate / Advanced / Job-Ready

    # Per-category breakdown {category: readiness %}
    category_breakdown:   dict = field(default_factory=dict)

    # High-importance missing skills (for priority recommendations)
    high_priority_missing: list = field(default_factory=list)
    medium_priority_missing: list = field(default_factory=list)
    low_priority_missing:  list = field(default_factory=list)

    # NLP similarity score (from TF-IDF module, optional)
    nlp_similarity_score: Optional[float] = None


# ── Readiness label ───────────────────────────────────────────────────────────

def get_readiness_label(readiness_score: float) -> str:
    """
    Map a numeric readiness score to a human-readable label.

    Thresholds:
        0 – 29   → Beginner
       30 – 49   → Developing
       50 – 69   → Intermediate
       70 – 84   → Advanced
       85 – 100  → Job-Ready
    """
    if readiness_score >= 85:
        return "Job-Ready"
    elif readiness_score >= 70:
        return "Advanced"
    elif readiness_score >= 50:
        return "Intermediate"
    elif readiness_score >= 30:
        return "Developing"
    else:
        return "Beginner"


# ── Core scoring function ────────────────────────────────────────────────────

def compute_scores(match_result: MatchResult,
                   nlp_similarity: Optional[float] = None) -> ScoreResult:
    """
    Compute all score metrics from a MatchResult object.

    Parameters
    ----------
    match_result : MatchResult
        Output of SkillMatcher.match()
    nlp_similarity : float, optional
        TF-IDF cosine similarity score from SkillExtractor (0–1).
        If provided, it is stored in the result for display purposes.

    Returns
    -------
    ScoreResult
    """
    matched  = match_result.matched
    partial  = match_result.partial
    missing  = match_result.missing
    all_reqs = matched + partial + missing

    total_required = len(all_reqs)
    n_matched      = len(matched)
    n_partial      = len(partial)
    n_missing      = len(missing)

    # ── 1. Skill Match Percentage (binary) ────────────────────────────────────
    skill_match_pct = round(n_matched / total_required * 100, 1) \
        if total_required > 0 else 0.0

    # ── 2. Weighted Readiness Score ───────────────────────────────────────────
    total_weight  = sum(s.weight for s in all_reqs)
    earned_weight = (
        sum(s.weight for s in matched)
        + sum(s.weight * PARTIAL_WEIGHT_FACTOR for s in partial)
    )

    readiness_score = round(earned_weight / total_weight * 100, 1) \
        if total_weight > 0 else 0.0

    # ── 3. Skill Gap Score ────────────────────────────────────────────────────
    gap_score = round(100.0 - readiness_score, 1)

    # ── 4. Readiness label ────────────────────────────────────────────────────
    label = get_readiness_label(readiness_score)

    # ── 5. Per-category breakdown ─────────────────────────────────────────────
    category_breakdown = _compute_category_breakdown(matched, partial, missing)

    # ── 6. Prioritize missing skills ──────────────────────────────────────────
    high_missing   = [s.skill for s in missing if s.importance == "high"]
    medium_missing = [s.skill for s in missing if s.importance == "medium"]
    low_missing    = [s.skill for s in missing if s.importance == "low"]

    return ScoreResult(
        target_role=match_result.target_role,
        total_required=total_required,
        n_matched=n_matched,
        n_partial=n_partial,
        n_missing=n_missing,
        skill_match_pct=skill_match_pct,
        readiness_score=readiness_score,
        gap_score=gap_score,
        readiness_label=label,
        category_breakdown=category_breakdown,
        high_priority_missing=high_missing,
        medium_priority_missing=medium_missing,
        low_priority_missing=low_missing,
        nlp_similarity_score=round(nlp_similarity * 100, 1)
            if nlp_similarity is not None else None,
    )


# ── Category breakdown helper ─────────────────────────────────────────────────

def _compute_category_breakdown(matched: list, partial: list,
                                missing: list) -> dict:
    """
    Compute readiness % per skill category.

    Returns
    -------
    dict : {category: {"readiness_pct": float, "matched": int,
                        "partial": int, "missing": int, "total": int}}
    """
    from collections import defaultdict

    category_data: dict = defaultdict(lambda: {
        "earned_weight": 0.0,
        "total_weight":  0.0,
        "matched": 0, "partial": 0, "missing": 0
    })

    for skill in matched:
        cat = skill.category
        category_data[cat]["earned_weight"] += skill.weight
        category_data[cat]["total_weight"]  += skill.weight
        category_data[cat]["matched"]       += 1

    for skill in partial:
        cat = skill.category
        category_data[cat]["earned_weight"] += skill.weight * PARTIAL_WEIGHT_FACTOR
        category_data[cat]["total_weight"]  += skill.weight
        category_data[cat]["partial"]       += 1

    for skill in missing:
        cat = skill.category
        category_data[cat]["total_weight"]  += skill.weight
        category_data[cat]["missing"]       += 1

    breakdown = {}
    for cat, data in category_data.items():
        total   = data["total_weight"]
        earned  = data["earned_weight"]
        pct     = round(earned / total * 100, 1) if total > 0 else 0.0
        n_total = data["matched"] + data["partial"] + data["missing"]
        breakdown[cat] = {
            "readiness_pct": pct,
            "matched": data["matched"],
            "partial": data["partial"],
            "missing": data["missing"],
            "total":   n_total,
        }

    return breakdown


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_score_result(score: ScoreResult) -> None:
    """Print a formatted scoring summary."""
    print(f"\n{'='*60}")
    print(f"  SKILL GAP ANALYSIS: {score.target_role}")
    print(f"{'='*60}")
    print(f"  Skills matched        : {score.n_matched} / {score.total_required}")
    print(f"  Partial matches       : {score.n_partial}")
    print(f"  Missing skills        : {score.n_missing}")
    print()
    print(f"  Skill Match %         : {score.skill_match_pct}%")
    print(f"  Weighted Readiness    : {score.readiness_score}%")
    print(f"  Skill Gap Score       : {score.gap_score}%")
    print(f"  Readiness Level       : {score.readiness_label}")
    if score.nlp_similarity_score is not None:
        print(f"  NLP Similarity Score  : {score.nlp_similarity_score}%")
    print()

    if score.category_breakdown:
        print("  Category Breakdown:")
        for cat, data in sorted(score.category_breakdown.items()):
            bar_len = int(data["readiness_pct"] / 5)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {cat:<20} [{bar}] {data['readiness_pct']:5.1f}%")
    print()

    if score.high_priority_missing:
        print(f"  🔴 HIGH priority missing ({len(score.high_priority_missing)}):")
        for s in score.high_priority_missing:
            print(f"       • {s}")

    if score.medium_priority_missing:
        print(f"  🟡 MEDIUM priority missing ({len(score.medium_priority_missing)}):")
        for s in score.medium_priority_missing:
            print(f"       • {s}")

    if score.low_priority_missing:
        print(f"  🟢 LOW priority missing ({len(score.low_priority_missing)}):")
        for s in score.low_priority_missing:
            print(f"       • {s}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from features.feature_engineering import build_role_profiles
    from matching.skill_matcher import SkillMatcher
    from preprocessing.data_cleaner import clean_dataset

    processed_csv = os.path.join(_ROOT, "data", "processed", "processed_roles.csv")
    if not os.path.exists(processed_csv):
        clean_dataset()

    df       = pd.read_csv(processed_csv)
    profiles = build_role_profiles(df)
    matcher  = SkillMatcher(role_profiles=profiles)

    user_skills = ["Python", "SQL", "Pandas", "Excel"]
    target_role = "Data Scientist"

    match_result = matcher.match(user_skills, target_role)
    score_result = compute_scores(match_result)
    print_score_result(score_result)
