"""
src/matching/skill_matcher.py

Stage 7 — Skill Matching Logic
--------------------------------
Compares a user's normalized skills against a target role's required skills
and classifies each required skill into one of three categories:

  MATCHED  — user has the skill (exact or alias-normalized match)
  PARTIAL  — user has a semantically similar but not identical skill
             (detected via TF-IDF cosine similarity above threshold)
  MISSING  — user does not have this skill at all

Why three categories?
  - Binary matched/missing ignores users who are "almost there" on a skill.
  - Partial matching gives more nuanced feedback and a fairer readiness score.
  - It helps recommendations focus on skills to deepen vs skills to start fresh.

Matching strategy (two-pass):
  Pass 1 — Exact matching: normalized skill strings are compared directly.
            Fast and deterministic.
  Pass 2 — Semantic matching (NLP): remaining unmatched required skills are
            compared to unmatched user skills using TF-IDF cosine similarity.
            Skills with similarity ≥ PARTIAL_THRESHOLD are marked PARTIAL.
            This handles aliases like "pytorch" ↔ "deep learning frameworks".
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from preprocessing.data_cleaner import normalize_user_skills, load_aliases

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALIASES_JSON = os.path.join(_ROOT, "data", "raw", "skill_aliases.json")

# Thresholds
PARTIAL_THRESHOLD = 0.45    # cosine sim above this = partial match
MATCH_THRESHOLD   = 0.85    # cosine sim above this = full match (NLP pass)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SkillMatch:
    """Represents the match result for a single required skill."""
    skill:       str
    importance:  str          # high / medium / low
    weight:      float        # numeric weight from importance
    category:    str
    status:      str          # "matched" | "partial" | "missing"
    matched_by:  str = ""     # "exact" | "nlp" | ""
    user_skill:  str = ""     # which user skill caused the match (if any)
    similarity:  float = 0.0  # cosine sim score (for NLP matches)


@dataclass
class MatchResult:
    """Full match result for a user vs target role comparison."""
    target_role:    str
    user_skills:    list
    matched:        list = field(default_factory=list)   # list of SkillMatch
    partial:        list = field(default_factory=list)
    missing:        list = field(default_factory=list)
    total_required: int  = 0

    @property
    def n_matched(self) -> int:
        return len(self.matched)

    @property
    def n_partial(self) -> int:
        return len(self.partial)

    @property
    def n_missing(self) -> int:
        return len(self.missing)

    @property
    def match_percentage(self) -> float:
        """Basic binary match percentage (matched / total × 100)."""
        if self.total_required == 0:
            return 0.0
        return round(self.n_matched / self.total_required * 100, 1)


# ── Skill Matcher ─────────────────────────────────────────────────────────────

class SkillMatcher:
    """
    Classifies required skills as matched / partial / missing
    for a given user skill set and target job role.

    Parameters
    ----------
    role_profiles : dict
        Output of feature_engineering.build_role_profiles()
    aliases_path : str
        Path to skill_aliases.json
    partial_threshold : float
        Cosine similarity threshold for partial match
    match_threshold : float
        Cosine similarity threshold for full NLP match
    """

    def __init__(self,
                 role_profiles: dict,
                 aliases_path: str = ALIASES_JSON,
                 partial_threshold: float = PARTIAL_THRESHOLD,
                 match_threshold: float = MATCH_THRESHOLD):
        self.role_profiles     = role_profiles
        self.aliases           = load_aliases(aliases_path)
        self.partial_threshold = partial_threshold
        self.match_threshold   = match_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def match(self, raw_user_skills: list, target_role: str) -> MatchResult:
        """
        Run the full two-pass matching pipeline.

        Parameters
        ----------
        raw_user_skills : list of str
            Skills as entered by the user (will be normalized internally)
        target_role : str
            Job role to compare against

        Returns
        -------
        MatchResult
        """
        if target_role not in self.role_profiles:
            raise ValueError(
                f"Role '{target_role}' not found in profiles. "
                f"Available: {list(self.role_profiles.keys())}"
            )

        # Normalize user inputs
        user_skills = normalize_user_skills(raw_user_skills, self.aliases)
        logger.info("Normalized user skills: %s", user_skills)

        # Fetch role requirements
        role_reqs = self.role_profiles[target_role]  # list of dicts

        result = MatchResult(
            target_role=target_role,
            user_skills=user_skills,
            total_required=len(role_reqs),
        )

        # ── Pass 1: Exact matching ─────────────────────────────────────────
        user_set          = set(user_skills)
        unmatched_reqs    = []
        unmatched_user    = list(user_skills)   # copy — consumed by pass 2

        for req in role_reqs:
            skill = req["skill"]
            if skill in user_set:
                result.matched.append(SkillMatch(
                    skill=skill,
                    importance=req["importance"],
                    weight=req["weight"],
                    category=req["category"],
                    status="matched",
                    matched_by="exact",
                    user_skill=skill,
                    similarity=1.0,
                ))
                if skill in unmatched_user:
                    unmatched_user.remove(skill)
            else:
                unmatched_reqs.append(req)

        logger.info("Pass 1 (exact): %d matched, %d unmatched required skills",
                    len(result.matched), len(unmatched_reqs))

        # ── Pass 2: NLP / semantic matching ───────────────────────────────
        if unmatched_reqs and unmatched_user:
            self._nlp_match(unmatched_reqs, unmatched_user, result)
        else:
            # All remaining unmatched requirements → missing
            for req in unmatched_reqs:
                result.missing.append(SkillMatch(
                    skill=req["skill"],
                    importance=req["importance"],
                    weight=req["weight"],
                    category=req["category"],
                    status="missing",
                ))

        return result

    # ── Pass 2: NLP matching ─────────────────────────────────────────────────

    def _nlp_match(self, unmatched_reqs: list,
                   unmatched_user: list,
                   result: MatchResult) -> None:
        """
        Use TF-IDF cosine similarity to find partial / full NLP matches
        between unmatched required skills and unmatched user skills.

        The TF-IDF vectorizer here is fitted on skill names (not job descriptions).
        This captures sub-word and phrase-level similarity between skill names.

        Example:
          "deep learning"   ↔ "tensorflow"  → moderate similarity
          "data visualisation" ↔ "data visualization" → high similarity
        """
        # Build a mini TF-IDF corpus over all skill name strings
        all_terms = unmatched_user + [req["skill"] for req in unmatched_reqs]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))

        try:
            tfidf_mat = vec.fit_transform(all_terms).toarray()
        except ValueError:
            # Fallback: all remaining unmatched reqs → missing
            for req in unmatched_reqs:
                result.missing.append(SkillMatch(
                    skill=req["skill"],
                    importance=req["importance"],
                    weight=req["weight"],
                    category=req["category"],
                    status="missing",
                ))
            return

        n_user = len(unmatched_user)
        user_vecs = tfidf_mat[:n_user]
        req_vecs  = tfidf_mat[n_user:]

        # Compute similarity matrix: required × user
        sim_matrix = cosine_similarity(req_vecs, user_vecs)

        used_user_indices = set()

        for r_idx, req in enumerate(unmatched_reqs):
            sims = sim_matrix[r_idx]
            best_u_idx = int(np.argmax(sims))
            best_sim   = float(sims[best_u_idx])

            if best_sim >= self.match_threshold and best_u_idx not in used_user_indices:
                # Full NLP match
                result.matched.append(SkillMatch(
                    skill=req["skill"],
                    importance=req["importance"],
                    weight=req["weight"],
                    category=req["category"],
                    status="matched",
                    matched_by="nlp",
                    user_skill=unmatched_user[best_u_idx],
                    similarity=round(best_sim, 4),
                ))
                used_user_indices.add(best_u_idx)

            elif best_sim >= self.partial_threshold and best_u_idx not in used_user_indices:
                # Partial NLP match
                result.partial.append(SkillMatch(
                    skill=req["skill"],
                    importance=req["importance"],
                    weight=req["weight"],
                    category=req["category"],
                    status="partial",
                    matched_by="nlp",
                    user_skill=unmatched_user[best_u_idx],
                    similarity=round(best_sim, 4),
                ))
                used_user_indices.add(best_u_idx)

            else:
                # No match found → missing
                result.missing.append(SkillMatch(
                    skill=req["skill"],
                    importance=req["importance"],
                    weight=req["weight"],
                    category=req["category"],
                    status="missing",
                ))

        logger.info("Pass 2 (NLP): %d additional matched, %d partial, total missing: %d",
                    sum(1 for m in result.matched if m.matched_by == "nlp"),
                    result.n_partial,
                    result.n_missing)


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_match_result(result: MatchResult) -> None:
    """Print a human-readable summary of a MatchResult."""
    print(f"\n{'='*60}")
    print(f"  Match Result: {result.target_role}")
    print(f"{'='*60}")
    print(f"  User skills    : {result.user_skills}")
    print(f"  Total required : {result.total_required}")
    print(f"  Matched        : {result.n_matched}")
    print(f"  Partial        : {result.n_partial}")
    print(f"  Missing        : {result.n_missing}")
    print(f"  Match %        : {result.match_percentage}%")

    print(f"\n  ✅ MATCHED SKILLS:")
    for s in result.matched:
        tag = f"(via {s.matched_by})" if s.matched_by else ""
        print(f"     {s.skill:<35} [{s.importance.upper()}] {tag}")

    print(f"\n  🟡 PARTIAL SKILLS:")
    for s in result.partial:
        print(f"     {s.skill:<35} [{s.importance.upper()}]  "
              f"(similar to '{s.user_skill}', sim={s.similarity})")

    print(f"\n  ❌ MISSING SKILLS:")
    for s in sorted(result.missing, key=lambda x: ["high", "medium", "low"].index(x.importance)):
        print(f"     {s.skill:<35} [{s.importance.upper()}]")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import pandas as pd
    sys.path.insert(0, os.path.join(_ROOT, "src"))
    from features.feature_engineering import build_role_profiles
    from preprocessing.data_cleaner import clean_dataset

    processed_csv = os.path.join(_ROOT, "data", "processed", "processed_roles.csv")
    if not os.path.exists(processed_csv):
        clean_dataset()

    df = pd.read_csv(processed_csv)
    profiles = build_role_profiles(df)
    matcher  = SkillMatcher(role_profiles=profiles)

    # Demo: user with Python, SQL, Pandas, Excel → Data Scientist
    user_skills  = ["Python", "SQL", "Pandas", "Excel"]
    target_role  = "Data Scientist"

    result = matcher.match(user_skills, target_role)
    print_match_result(result)
