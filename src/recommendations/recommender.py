"""
src/recommendations/recommender.py

Stage 9 — Recommendation Engine
----------------------------------
Generates a personalized, prioritized learning path from a ScoreResult.

How prioritization works:
  1. HIGH importance missing skills → learn first (biggest impact on readiness)
  2. PARTIAL skills → deepen next (user has some knowledge, needs to improve)
  3. MEDIUM importance missing skills → learn after high-priority gaps
  4. LOW importance missing skills → optional / nice-to-have

Within each tier, skills are further sorted by:
  - Cross-role demand (skills needed in many roles come first — more transferable)
  - Availability of learning resources

Each recommended skill includes:
  - Priority rank
  - Importance tier
  - Why it is needed (role context)
  - Curated learning resources with URLs, type, and estimated duration

NLP integration:
  The recommender also accepts the TF-IDF user-role similarity scores to
  suggest alternative roles the user might be closer to achieving, helping
  career pivots.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from scoring.score_engine import ScoreResult
from matching.skill_matcher import MatchResult

RESOURCES_JSON = os.path.join(_ROOT, "data", "raw", "learning_resources.json")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class LearningResource:
    """A single learning resource for a skill."""
    title:    str
    url:      str
    type:     str        # course / book / tutorial / free / interactive / docs
    duration: str


@dataclass
class SkillRecommendation:
    """A recommended skill to learn, with context and resources."""
    rank:         int
    skill:        str
    priority:     str        # "high" | "medium" | "low" | "deepen"
    reason:       str
    resources:    list       # list of LearningResource
    description:  str = ""


@dataclass
class LearningPath:
    """Complete personalized learning path returned to the user."""
    target_role:          str
    readiness_score:      float
    readiness_label:      str
    total_skills_to_learn: int
    recommendations:      list = field(default_factory=list)   # SkillRecommendation
    alternative_roles:    list = field(default_factory=list)   # (role, similarity)
    summary_message:      str  = ""


# ── Recommender ───────────────────────────────────────────────────────────────

class SkillRecommender:
    """
    Generates a prioritized learning path from scoring and matching results.

    Parameters
    ----------
    resources_path : str
        Path to learning_resources.json
    """

    def __init__(self, resources_path: str = RESOURCES_JSON):
        self.resources = self._load_resources(resources_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(self,
                  match_result: MatchResult,
                  score_result: ScoreResult,
                  nlp_similarities: Optional[dict] = None) -> LearningPath:
        """
        Build the full learning path.

        Parameters
        ----------
        match_result : MatchResult
            Skill matching output from Stage 7
        score_result : ScoreResult
            Scoring output from Stage 8
        nlp_similarities : dict, optional
            {role: similarity_score} from the TF-IDF module (Stage 6)

        Returns
        -------
        LearningPath
        """
        recommendations = []
        rank = 1

        # ── Phase 1: HIGH importance missing skills ───────────────────────────
        for skill_match in match_result.missing:
            if skill_match.importance == "high":
                rec = self._make_recommendation(
                    rank=rank,
                    skill=skill_match.skill,
                    priority="high",
                    reason=(f"This is a CRITICAL skill for {score_result.target_role}. "
                            f"Acquiring it will have the highest impact on your readiness score."),
                )
                recommendations.append(rec)
                rank += 1

        # ── Phase 2: Partial skills (deepen existing knowledge) ───────────────
        for skill_match in match_result.partial:
            rec = self._make_recommendation(
                rank=rank,
                skill=skill_match.skill,
                priority="deepen",
                reason=(f"You have some knowledge of '{skill_match.user_skill}', "
                        f"but '{skill_match.skill}' is required at a higher level "
                        f"for {score_result.target_role}. Deepen this skill next."),
            )
            recommendations.append(rec)
            rank += 1

        # ── Phase 3: MEDIUM importance missing skills ─────────────────────────
        for skill_match in match_result.missing:
            if skill_match.importance == "medium":
                rec = self._make_recommendation(
                    rank=rank,
                    skill=skill_match.skill,
                    priority="medium",
                    reason=(f"This is an IMPORTANT skill for {score_result.target_role}. "
                            f"Learn this after covering high-priority gaps."),
                )
                recommendations.append(rec)
                rank += 1

        # ── Phase 4: LOW importance missing skills ────────────────────────────
        for skill_match in match_result.missing:
            if skill_match.importance == "low":
                rec = self._make_recommendation(
                    rank=rank,
                    skill=skill_match.skill,
                    priority="low",
                    reason=(f"Nice-to-have skill for {score_result.target_role}. "
                            f"Learn when high and medium gaps are covered."),
                )
                recommendations.append(rec)
                rank += 1

        # ── Alternative role suggestions from NLP ──────────────────────────────
        alt_roles = []
        if nlp_similarities:
            sorted_roles = sorted(
                [(r, s) for r, s in nlp_similarities.items()
                 if r != score_result.target_role],
                key=lambda x: -x[1]
            )
            alt_roles = sorted_roles[:3]    # top 3 alternative roles

        # ── Summary message ────────────────────────────────────────────────────
        message = self._build_summary(score_result, len(recommendations))

        return LearningPath(
            target_role=score_result.target_role,
            readiness_score=score_result.readiness_score,
            readiness_label=score_result.readiness_label,
            total_skills_to_learn=len(recommendations),
            recommendations=recommendations,
            alternative_roles=alt_roles,
            summary_message=message,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_recommendation(self, rank: int, skill: str,
                              priority: str, reason: str) -> SkillRecommendation:
        """Build a SkillRecommendation for a given skill."""
        resource_data = self.resources.get(skill, {})
        description   = resource_data.get("description", "")
        raw_resources = resource_data.get("resources", [])

        resources = [
            LearningResource(
                title=r.get("title", ""),
                url=r.get("url", "#"),
                type=r.get("type", ""),
                duration=r.get("duration", ""),
            )
            for r in raw_resources
        ]

        # If no curated resources, provide a generic search suggestion
        if not resources:
            resources = [LearningResource(
                title=f"Search: '{skill}' tutorial",
                url=f"https://www.google.com/search?q={skill.replace(' ', '+')}+tutorial",
                type="search",
                duration="varies",
            )]

        return SkillRecommendation(
            rank=rank,
            skill=skill,
            priority=priority,
            reason=reason,
            resources=resources,
            description=description,
        )

    def _build_summary(self, score: ScoreResult, n_recs: int) -> str:
        """Generate a personalized summary message."""
        if score.readiness_score >= 85:
            return (f"🚀 You are nearly job-ready for {score.target_role}! "
                    f"Focus on the {n_recs} remaining skills to reach 100%.")
        elif score.readiness_score >= 70:
            return (f"⭐ Strong profile for {score.target_role}. "
                    f"Filling {n_recs} skill gaps will make you highly competitive.")
        elif score.readiness_score >= 50:
            return (f"📈 Good foundation for {score.target_role}. "
                    f"You need to learn {n_recs} more skills — focus on high-priority ones first.")
        elif score.readiness_score >= 30:
            return (f"🌱 You're on the right track for {score.target_role}. "
                    f"Commit to the {n_recs}-skill learning path to build your profile.")
        else:
            return (f"🔰 Starting your journey toward {score.target_role}. "
                    f"Begin with the high-priority skills — {len(score.high_priority_missing)} "
                    f"foundational skills will give you the biggest boost.")

    @staticmethod
    def _load_resources(path: str) -> dict:
        if not os.path.exists(path):
            logger.warning("Learning resources file not found: %s", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_learning_path(path: LearningPath) -> None:
    """Print a formatted learning path to the console."""
    print(f"\n{'='*65}")
    print(f"  PERSONALIZED LEARNING PATH: {path.target_role}")
    print(f"{'='*65}")
    print(f"  Readiness    : {path.readiness_score}% — {path.readiness_label}")
    print(f"  Skills to learn : {path.total_skills_to_learn}")
    print(f"\n  {path.summary_message}")

    priority_order = ["high", "deepen", "medium", "low"]
    priority_labels = {
        "high":   "🔴 HIGH PRIORITY (Missing Critical Skills)",
        "deepen": "🟡 DEEPEN EXISTING SKILLS",
        "medium": "🟠 MEDIUM PRIORITY",
        "low":    "🟢 LOW PRIORITY (Nice-to-Have)",
    }

    for p in priority_order:
        recs = [r for r in path.recommendations if r.priority == p]
        if not recs:
            continue
        print(f"\n  {priority_labels[p]}")
        print(f"  {'─'*55}")
        for rec in recs:
            print(f"\n  {rec.rank}. {rec.skill.upper()}")
            if rec.description:
                print(f"     {rec.description}")
            print(f"     Reason: {rec.reason}")
            print(f"     Resources:")
            for res in rec.resources[:2]:   # show top 2 resources
                print(f"       → [{res.type.upper()}] {res.title}")
                print(f"          {res.url}  ({res.duration})")

    if path.alternative_roles:
        print(f"\n  💡 ALTERNATIVE ROLES YOU MIGHT ALSO CONSIDER:")
        for role, score in path.alternative_roles:
            print(f"     • {role:<35} (NLP similarity: {score:.2%})")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from features.feature_engineering import build_role_profiles
    from matching.skill_matcher import SkillMatcher
    from scoring.score_engine import compute_scores
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

    recommender  = SkillRecommender()
    path         = recommender.recommend(match_result, score_result)
    print_learning_path(path)
