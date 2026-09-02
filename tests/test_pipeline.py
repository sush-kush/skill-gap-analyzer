"""
tests/test_pipeline.py

Stage 11 — Integration & Unit Tests
--------------------------------------
Tests for every major module in the pipeline.
Run with:  python -m pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    """Load the processed dataset (runs data_cleaner if needed)."""
    from preprocessing.data_cleaner import clean_dataset
    processed_csv = os.path.join(_ROOT, "data", "processed", "processed_roles.csv")
    if not os.path.exists(processed_csv):
        clean_dataset()
    return pd.read_csv(processed_csv)


@pytest.fixture(scope="module")
def role_profiles(sample_df):
    from features.feature_engineering import build_role_profiles
    return build_role_profiles(sample_df)


@pytest.fixture(scope="module")
def matcher(role_profiles):
    from matching.skill_matcher import SkillMatcher
    return SkillMatcher(role_profiles=role_profiles)


# ─────────────────────────────────────────────
# Stage 3 — Data Cleaning
# ─────────────────────────────────────────────

class TestDataCleaner:

    def test_normalize_skill_lowercase(self):
        from preprocessing.data_cleaner import normalize_skill
        assert normalize_skill("Python", {}) == "python"

    def test_normalize_skill_alias(self):
        from preprocessing.data_cleaner import normalize_skill
        aliases = {"ml": "machine learning", "sklearn": "scikit-learn"}
        assert normalize_skill("ml", aliases) == "machine learning"
        assert normalize_skill("sklearn", aliases) == "scikit-learn"

    def test_normalize_skill_strips_whitespace(self):
        from preprocessing.data_cleaner import normalize_skill
        assert normalize_skill("  Python  ", {}) == "python"

    def test_normalize_user_skills_deduplicates(self):
        from preprocessing.data_cleaner import normalize_user_skills, load_aliases
        aliases = load_aliases(os.path.join(_ROOT, "data", "raw", "skill_aliases.json"))
        result = normalize_user_skills(["Python", "python", "PYTHON"], aliases)
        assert result.count("python") == 1

    def test_processed_csv_has_required_columns(self, sample_df):
        for col in ["role", "skill", "importance", "category"]:
            assert col in sample_df.columns

    def test_importance_values_valid(self, sample_df):
        valid = {"high", "medium", "low"}
        assert set(sample_df["importance"].unique()).issubset(valid)

    def test_no_null_skills(self, sample_df):
        assert sample_df["skill"].isnull().sum() == 0

    def test_no_null_roles(self, sample_df):
        assert sample_df["role"].isnull().sum() == 0

    def test_no_duplicate_role_skill_pairs(self, sample_df):
        dupes = sample_df.duplicated(subset=["role", "skill"]).sum()
        assert dupes == 0


# ─────────────────────────────────────────────
# Stage 5 — Feature Engineering
# ─────────────────────────────────────────────

class TestFeatureEngineering:

    def test_vocabulary_not_empty(self, sample_df):
        from features.feature_engineering import build_skill_vocabulary
        vocab = build_skill_vocabulary(sample_df)
        assert len(vocab) > 0

    def test_vocabulary_sorted(self, sample_df):
        from features.feature_engineering import build_skill_vocabulary
        vocab = build_skill_vocabulary(sample_df)
        assert vocab == sorted(vocab)

    def test_binary_matrix_shape(self, sample_df):
        from features.feature_engineering import build_skill_vocabulary, build_binary_matrix
        vocab  = build_skill_vocabulary(sample_df)
        matrix = build_binary_matrix(sample_df, vocab)
        assert matrix.shape[0] == sample_df["role"].nunique()
        assert matrix.shape[1] == len(vocab)

    def test_binary_matrix_values_01(self, sample_df):
        from features.feature_engineering import build_skill_vocabulary, build_binary_matrix
        vocab  = build_skill_vocabulary(sample_df)
        matrix = build_binary_matrix(sample_df, vocab)
        assert set(matrix.values.flatten().tolist()).issubset({0, 1})

    def test_weighted_matrix_values_in_range(self, sample_df):
        from features.feature_engineering import build_skill_vocabulary, build_weighted_matrix
        vocab  = build_skill_vocabulary(sample_df)
        matrix = build_weighted_matrix(sample_df, vocab)
        vals = matrix.values.flatten()
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_user_vector_length(self, sample_df):
        from features.feature_engineering import build_skill_vocabulary, build_user_vector
        vocab   = build_skill_vocabulary(sample_df)
        vec     = build_user_vector(["python", "sql"], vocab)
        assert len(vec) == len(vocab)

    def test_user_vector_known_skill_nonzero(self, sample_df):
        from features.feature_engineering import build_skill_vocabulary, build_user_vector
        vocab = build_skill_vocabulary(sample_df)
        vec   = build_user_vector(["python"], vocab)
        idx   = vocab.index("python")
        assert vec[idx] == 1.0

    def test_cosine_similarity_identical(self):
        import numpy as np
        from features.feature_engineering import cosine_similarity_vectors
        v = np.array([0.5, 0.3, 0.8])
        assert abs(cosine_similarity_vectors(v, v) - 1.0) < 1e-9

    def test_cosine_similarity_zero_vector(self):
        import numpy as np
        from features.feature_engineering import cosine_similarity_vectors
        v = np.array([1.0, 0.5])
        z = np.array([0.0, 0.0])
        assert cosine_similarity_vectors(v, z) == 0.0

    def test_role_profiles_all_roles_present(self, sample_df, role_profiles):
        expected_roles = set(sample_df["role"].unique())
        assert expected_roles == set(role_profiles.keys())

    def test_role_profile_has_required_keys(self, role_profiles):
        for role, skills in role_profiles.items():
            for s in skills:
                assert "skill" in s
                assert "importance" in s
                assert "weight" in s
                assert "category" in s


# ─────────────────────────────────────────────
# Stage 6 — NLP / TF-IDF
# ─────────────────────────────────────────────

class TestSkillExtractor:

    def test_extractor_fits_without_error(self):
        from nlp.skill_extractor import SkillExtractor, load_job_descriptions
        jd = load_job_descriptions()
        ext = SkillExtractor()
        ext.fit(jd)
        assert ext._is_fitted

    def test_vocabulary_populated_after_fit(self):
        from nlp.skill_extractor import SkillExtractor, load_job_descriptions
        jd = load_job_descriptions()
        ext = SkillExtractor()
        ext.fit(jd)
        assert len(ext.vocabulary_) > 0

    def test_user_role_similarity_returns_all_roles(self):
        from nlp.skill_extractor import SkillExtractor, load_job_descriptions
        jd = load_job_descriptions()
        ext = SkillExtractor()
        ext.fit(jd)
        sims = ext.user_role_similarity(["python", "machine learning"])
        assert set(sims.keys()) == set(jd.keys())

    def test_similarity_values_in_range(self):
        from nlp.skill_extractor import SkillExtractor, load_job_descriptions
        jd = load_job_descriptions()
        ext = SkillExtractor()
        ext.fit(jd)
        sims = ext.user_role_similarity(["python", "sql"])
        for score in sims.values():
            assert 0.0 <= score <= 1.0

    def test_not_fitted_raises_error(self):
        from nlp.skill_extractor import SkillExtractor
        ext = SkillExtractor()
        with pytest.raises(RuntimeError):
            ext.user_role_similarity(["python"])


# ─────────────────────────────────────────────
# Stage 7 — Skill Matching
# ─────────────────────────────────────────────

class TestSkillMatcher:

    def test_match_returns_match_result(self, matcher):
        from matching.skill_matcher import MatchResult
        result = matcher.match(["Python", "SQL", "Pandas"], "Data Scientist")
        assert isinstance(result, MatchResult)

    def test_total_required_correct(self, matcher, sample_df):
        target_role    = "Data Scientist"
        expected_total = len(sample_df[sample_df["role"] == target_role])
        result         = matcher.match(["Python"], target_role)
        assert result.total_required == expected_total

    def test_python_matched_for_data_scientist(self, matcher):
        result = matcher.match(["python"], "Data Scientist")
        matched_skills = [s.skill for s in result.matched]
        assert "python" in matched_skills

    def test_n_matched_plus_partial_plus_missing_equals_total(self, matcher):
        result = matcher.match(["Python", "SQL"], "Data Analyst")
        assert result.n_matched + result.n_partial + result.n_missing == result.total_required

    def test_empty_user_skills(self, matcher):
        result = matcher.match([], "Data Scientist")
        assert result.n_matched == 0
        assert result.n_missing == result.total_required

    def test_invalid_role_raises(self, matcher):
        with pytest.raises(ValueError):
            matcher.match(["python"], "Astronaut")

    def test_match_percentage_range(self, matcher):
        result = matcher.match(["Python", "SQL", "Pandas"], "Data Analyst")
        assert 0.0 <= result.match_percentage <= 100.0


# ─────────────────────────────────────────────
# Stage 8 — Scoring Engine
# ─────────────────────────────────────────────

class TestScoringEngine:

    def test_readiness_score_range(self, matcher):
        from scoring.score_engine import compute_scores
        result = matcher.match(["Python", "SQL"], "Data Scientist")
        score  = compute_scores(result)
        assert 0.0 <= score.readiness_score <= 100.0

    def test_gap_score_equals_100_minus_readiness(self, matcher):
        from scoring.score_engine import compute_scores
        result = matcher.match(["Python"], "Data Scientist")
        score  = compute_scores(result)
        assert abs(score.readiness_score + score.gap_score - 100.0) < 0.1

    def test_all_skills_matched_gives_100(self, matcher, sample_df):
        from scoring.score_engine import compute_scores
        target     = "Data Analyst"
        all_skills = sample_df[sample_df["role"] == target]["skill"].tolist()
        result     = matcher.match(all_skills, target)
        score      = compute_scores(result)
        assert score.readiness_score == 100.0

    def test_no_skills_gives_0(self, matcher):
        from scoring.score_engine import compute_scores
        result = matcher.match([], "Data Scientist")
        score  = compute_scores(result)
        assert score.readiness_score == 0.0

    def test_readiness_label_assigned(self, matcher):
        from scoring.score_engine import compute_scores
        result = matcher.match(["Python"], "Python Developer")
        score  = compute_scores(result)
        assert score.readiness_label != ""

    def test_high_priority_missing_are_high_importance(self, matcher):
        from scoring.score_engine import compute_scores
        result = matcher.match([], "Data Scientist")
        score  = compute_scores(result)
        for skill in score.high_priority_missing:
            assert isinstance(skill, str)

    def test_category_breakdown_has_categories(self, matcher):
        from scoring.score_engine import compute_scores
        result = matcher.match(["Python"], "Data Scientist")
        score  = compute_scores(result)
        assert len(score.category_breakdown) > 0


# ─────────────────────────────────────────────
# Stage 9 — Recommendations
# ─────────────────────────────────────────────

class TestRecommender:

    def test_recommendations_produced(self, matcher):
        from scoring.score_engine import compute_scores
        from recommendations.recommender import SkillRecommender
        result  = matcher.match(["Python", "SQL"], "Data Scientist")
        score   = compute_scores(result)
        rec     = SkillRecommender()
        path    = rec.recommend(result, score)
        assert len(path.recommendations) > 0

    def test_recommendations_ranked_sequentially(self, matcher):
        from scoring.score_engine import compute_scores
        from recommendations.recommender import SkillRecommender
        result  = matcher.match([], "Data Analyst")
        score   = compute_scores(result)
        rec     = SkillRecommender()
        path    = rec.recommend(result, score)
        ranks   = [r.rank for r in path.recommendations]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_no_recommendations_when_all_matched(self, matcher, sample_df):
        from scoring.score_engine import compute_scores
        from recommendations.recommender import SkillRecommender
        target     = "Data Analyst"
        all_skills = sample_df[sample_df["role"] == target]["skill"].tolist()
        result     = matcher.match(all_skills, target)
        score      = compute_scores(result)
        rec        = SkillRecommender()
        path       = rec.recommend(result, score)
        assert len(path.recommendations) == 0

    def test_summary_message_not_empty(self, matcher):
        from scoring.score_engine import compute_scores
        from recommendations.recommender import SkillRecommender
        result = matcher.match(["Python"], "Data Scientist")
        score  = compute_scores(result)
        rec    = SkillRecommender()
        path   = rec.recommend(result, score)
        assert path.summary_message != ""

    def test_each_recommendation_has_resources(self, matcher):
        from scoring.score_engine import compute_scores
        from recommendations.recommender import SkillRecommender
        result = matcher.match([], "Python Developer")
        score  = compute_scores(result)
        rec    = SkillRecommender()
        path   = rec.recommend(result, score)
        for r in path.recommendations:
            assert len(r.resources) >= 1


# ─────────────────────────────────────────────
# End-to-end integration test
# ─────────────────────────────────────────────

class TestEndToEnd:

    def test_full_pipeline_runs(self):
        """
        Full pipeline: clean → profile → match → score → recommend.
        Uses the sample from the project spec: Python, SQL, Pandas, Excel → Data Scientist.
        """
        from preprocessing.data_cleaner import clean_dataset
        from features.feature_engineering import build_role_profiles
        from matching.skill_matcher import SkillMatcher
        from scoring.score_engine import compute_scores
        from recommendations.recommender import SkillRecommender

        processed_csv = os.path.join(_ROOT, "data", "processed", "processed_roles.csv")
        if not os.path.exists(processed_csv):
            clean_dataset()

        df       = pd.read_csv(processed_csv)
        profiles = build_role_profiles(df)
        matcher  = SkillMatcher(role_profiles=profiles)

        user_skills = ["Python", "SQL", "Pandas", "Excel"]
        target_role = "Data Scientist"

        match_result  = matcher.match(user_skills, target_role)
        score_result  = compute_scores(match_result)
        recommender   = SkillRecommender()
        learning_path = recommender.recommend(match_result, score_result)

        # Python and SQL and Pandas should be matched
        matched_names = [s.skill for s in match_result.matched]
        assert "python" in matched_names
        assert "sql"    in matched_names
        assert "pandas" in matched_names

        # Readiness should be > 0 and < 100 (user has some but not all skills)
        assert 0 < score_result.readiness_score < 100

        # Recommendations must exist (many skills missing)
        assert len(learning_path.recommendations) > 0

        # Gap + readiness = 100
        assert abs(score_result.readiness_score + score_result.gap_score - 100.0) < 0.1
