"""
src/nlp/skill_extractor.py

Stage 6 — NLP Pipeline: TF-IDF Skill Extraction
-------------------------------------------------
This module is the core ML/NLP component of the project.

What it does:
  1. Loads raw job descriptions (unstructured text per role)
  2. Preprocesses text — tokenization, stopword removal, normalization
  3. Trains a TF-IDF vectorizer over all job descriptions
  4. Extracts the highest-scoring skill-like n-grams per description
  5. Cross-references extracted terms against the known skill vocabulary
     to identify confirmed skills — this is the NLP-based skill detection
  6. Also computes TF-IDF-based cosine similarity between a user's skill
     text and each role's job description (user-to-role similarity scoring)
  7. Saves the fitted vectorizer for later use

Why TF-IDF + cosine similarity:
  - TF-IDF (Term Frequency–Inverse Document Frequency) weights terms that
    are important within a specific role's description but rare overall.
    This highlights role-specific skills rather than common filler words.
  - Cosine similarity measures the angle between two TF-IDF vectors,
    giving a meaningful 0–1 similarity score regardless of text length.
  - This is a genuine NLP technique, not a rule-based keyword lookup.
  - It handles slight wording variations better than exact string matching.

ML component summary:
  Algorithm   : TF-IDF vectorization (sklearn TfidfVectorizer)
  Input       : Raw job description text (one document per role)
  Features    : Unigram + bigram TF-IDF weights
  Extraction  : Top-scoring terms filtered against known skill vocabulary
  Similarity  : Cosine similarity between user skill text and role JD vector
  Training    : Unsupervised — fit on the job descriptions corpus
  Prediction  : transform() on new user skill text → similarity scores
"""

import os
import re
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Path resolution ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

JOB_DESCRIPTIONS_JSON = os.path.join(_ROOT, "data", "raw", "job_descriptions.json")
ALIASES_JSON          = os.path.join(_ROOT, "data", "raw", "skill_aliases.json")
PROCESSED_DIR         = os.path.join(_ROOT, "data", "processed")
MODELS_DIR            = os.path.join(_ROOT, "models")
VECTORIZER_PATH       = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard English stopwords relevant to job descriptions
# (kept minimal — we use sklearn's built-in list + these extras)
EXTRA_STOPWORDS = {
    "experience", "knowledge", "understanding", "ability", "strong",
    "good", "excellent", "required", "preferred", "responsible",
    "work", "working", "use", "using", "used", "role", "team",
    "skill", "skills", "proficiency", "proficient", "familiar",
    "familiarity", "including", "also", "well", "plus", "basic",
    "advanced", "must", "will", "candidate", "candidates",
    "expected", "key", "core", "critical", "important", "essential",
}


# ── Text preprocessing ───────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalize a job description text for TF-IDF.

    Steps:
      - Lowercase
      - Remove punctuation (keep hyphens and slashes for "ci/cd", "deep-learning")
      - Normalize whitespace

    Parameters
    ----------
    text : str
        Raw job description paragraph

    Returns
    -------
    str
        Cleaned text
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\-/]", " ", text)   # keep word chars, spaces, - and /
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── TF-IDF Vectorizer ─────────────────────────────────────────────────────────

class SkillExtractor:
    """
    NLP component: trains a TF-IDF model on job descriptions and provides
    skill extraction and user-to-role similarity scoring.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Fitted sklearn TF-IDF vectorizer
    role_names : list of str
        Ordered list of role names (rows of the TF-IDF matrix)
    tfidf_matrix : np.ndarray
        TF-IDF matrix (n_roles × n_features)
    vocabulary_ : list of str
        Feature names from the vectorizer
    """

    def __init__(self,
                 max_features: int = 500,
                 ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            preprocessor=preprocess_text,
            token_pattern=r"[a-zA-Z][a-zA-Z0-9\-/]{1,}",  # min 2 chars
            sublinear_tf=True,   # use 1 + log(tf) to reduce impact of very common terms
        )
        self.role_names:   list      = []
        self.tfidf_matrix: np.ndarray = None
        self.vocabulary_:  list      = []
        self._is_fitted:   bool      = False

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, job_descriptions: dict) -> "SkillExtractor":
        """
        Fit the TF-IDF vectorizer on all job descriptions.

        Parameters
        ----------
        job_descriptions : dict
            {role_name: description_text}

        Returns
        -------
        self
        """
        self.role_names = list(job_descriptions.keys())
        corpus = [job_descriptions[role] for role in self.role_names]

        logger.info("Fitting TF-IDF on %d job descriptions...", len(corpus))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()
        self.vocabulary_ = self.vectorizer.get_feature_names_out().tolist()
        self._is_fitted = True

        logger.info("TF-IDF vocabulary size: %d terms", len(self.vocabulary_))
        return self

    # ── Skill extraction ─────────────────────────────────────────────────────

    def extract_skills_for_role(self,
                                role: str,
                                known_skills: list,
                                top_n: int = 30) -> list:
        """
        Extract the most relevant skills from a role's job description
        by selecting top TF-IDF scoring terms that match the known skill vocabulary.

        This is the NLP-based skill identification step:
        Rather than hardcoding which skills belong to a role, we let the
        TF-IDF scores tell us which terms are statistically important in
        that role's description — then we filter to known skill terms.

        Parameters
        ----------
        role : str
            Job role name
        known_skills : list of str
            Known skill vocabulary (from feature_engineering.py)
        top_n : int
            Number of top TF-IDF terms to consider before filtering

        Returns
        -------
        list of str
            Extracted skills (subset of known_skills) sorted by TF-IDF score
        """
        self._assert_fitted()
        if role not in self.role_names:
            logger.warning("Role '%s' not in fitted corpus. Returning empty list.", role)
            return []

        role_idx = self.role_names.index(role)
        role_vec = self.tfidf_matrix[role_idx]

        # Get top-N terms by TF-IDF score
        top_indices = np.argsort(role_vec)[::-1][:top_n]
        top_terms = [(self.vocabulary_[i], role_vec[i]) for i in top_indices
                     if role_vec[i] > 0]

        # Filter to terms that match our known skill vocabulary
        known_set = set(known_skills)
        extracted = [term for term, score in top_terms if term in known_set]

        return extracted

    # ── User ↔ Role similarity ────────────────────────────────────────────────

    def user_role_similarity(self, user_skills: list) -> dict:
        """
        Compute cosine similarity between the user's skill text and
        every role's job description TF-IDF vector.

        The user's skills are joined into a text string and transformed
        using the already-fitted vectorizer — then cosine similarity is
        computed against all role vectors.

        This gives a data-driven similarity score (0–1) per role,
        reflecting how closely the user's skill profile aligns with
        each job description's vocabulary.

        Parameters
        ----------
        user_skills : list of str
            Normalized user skills

        Returns
        -------
        dict
            {role_name: similarity_score (float 0–1)}
        """
        self._assert_fitted()

        # Join skills as a pseudo-document
        user_text = " ".join(user_skills)
        user_vec = self.vectorizer.transform([user_text]).toarray()

        sims = cosine_similarity(user_vec, self.tfidf_matrix)[0]

        return {role: float(round(sims[i], 4))
                for i, role in enumerate(self.role_names)}

    # ── Top n-grams for a role ────────────────────────────────────────────────

    def get_top_terms(self, role: str, top_n: int = 20) -> list:
        """
        Return the top TF-IDF terms (with scores) for a given role.
        Useful for debugging and understanding what the model has learned.

        Returns
        -------
        list of (term, score) tuples
        """
        self._assert_fitted()
        if role not in self.role_names:
            return []

        role_idx = self.role_names.index(role)
        role_vec = self.tfidf_matrix[role_idx]
        top_indices = np.argsort(role_vec)[::-1][:top_n]
        return [(self.vocabulary_[i], round(float(role_vec[i]), 4))
                for i in top_indices if role_vec[i] > 0]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = VECTORIZER_PATH) -> None:
        """Serialize the fitted extractor to disk (pickle)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("SkillExtractor saved to: %s", path)

    @classmethod
    def load(cls, path: str = VECTORIZER_PATH) -> "SkillExtractor":
        """Deserialize a previously saved SkillExtractor from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("SkillExtractor loaded from: %s", path)
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "SkillExtractor has not been fitted yet. Call .fit(job_descriptions) first."
            )


# ── Convenience functions ─────────────────────────────────────────────────────

def load_job_descriptions(path: str = JOB_DESCRIPTIONS_JSON) -> dict:
    """Load the raw job descriptions dictionary from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_and_save_extractor(jd_path: str = JOB_DESCRIPTIONS_JSON,
                             save_path: str = VECTORIZER_PATH) -> SkillExtractor:
    """
    Full pipeline: load JDs → fit TF-IDF → save model.

    Returns
    -------
    SkillExtractor
        Fitted extractor
    """
    job_descriptions = load_job_descriptions(jd_path)
    extractor = SkillExtractor(max_features=500, ngram_range=(1, 2))
    extractor.fit(job_descriptions)
    extractor.save(save_path)
    return extractor


def get_extractor(force_rebuild: bool = False,
                  jd_path: str = JOB_DESCRIPTIONS_JSON,
                  save_path: str = VECTORIZER_PATH) -> SkillExtractor:
    """
    Load extractor from disk if available, otherwise build and save it.

    Parameters
    ----------
    force_rebuild : bool
        If True, always retrain even if a saved model exists
    jd_path : str
        Path to job_descriptions.json (override for cloud deployments)
    save_path : str
        Path where the fitted model will be saved/loaded

    Returns
    -------
    SkillExtractor
    """
    if not force_rebuild and os.path.exists(save_path):
        return SkillExtractor.load(save_path)
    return build_and_save_extractor(jd_path=jd_path, save_path=save_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(_ROOT, "src"))
    from preprocessing.data_cleaner import clean_dataset

    # Step 1: ensure processed dataset exists
    processed_csv = os.path.join(PROCESSED_DIR, "processed_roles.csv")
    if not os.path.exists(processed_csv):
        clean_dataset()
    df = pd.read_csv(processed_csv)
    known_skills = sorted(df["skill"].unique().tolist())

    # Step 2: fit extractor
    print("Building TF-IDF skill extractor...")
    extractor = build_and_save_extractor()

    # Step 3: show top terms per role
    print("\n=== Top TF-IDF Terms per Role ===")
    for role in extractor.role_names:
        terms = extractor.get_top_terms(role, top_n=8)
        print(f"\n  {role}:")
        for term, score in terms:
            print(f"    {term:<30} {score}")

    # Step 4: extract skills from JD for Data Scientist
    print("\n=== NLP-Extracted Skills for 'Data Scientist' ===")
    extracted = extractor.extract_skills_for_role("Data Scientist", known_skills)
    print(extracted)

    # Step 5: user-to-role similarity
    sample_skills = ["python", "sql", "pandas", "machine learning"]
    print(f"\n=== User Similarity Scores for skills: {sample_skills} ===")
    sims = extractor.user_role_similarity(sample_skills)
    for role, score in sorted(sims.items(), key=lambda x: -x[1]):
        print(f"  {role:<30} {score:.4f}")
