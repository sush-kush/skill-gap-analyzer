"""
src/features/feature_engineering.py

Stage 5 — Feature Engineering
-------------------------------
Transforms the cleaned role-skill dataset into feature representations
that downstream ML/NLP modules can consume.

What is built here:
  1. Role profiles   — dict mapping each role → list of (skill, importance, category)
  2. Skill vocabulary — sorted list of all unique skills (used as feature columns)
  3. Binary skill matrix — one-hot matrix: roles × skills (1 = required, 0 = not)
  4. Weighted skill matrix — same matrix but cells use importance weight values
  5. User skill vector — converts a user's skill list into a binary/weighted vector

Why feature engineering:
  Raw strings can't be fed into numerical ML/similarity computations.
  Converting skills to vectors enables cosine similarity comparisons between
  a user profile and a role profile — the core of the gap analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional

# ── Resolve paths ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

PROCESSED_CSV = os.path.join(_ROOT, "data", "processed", "processed_roles.csv")
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")

# Importance → numeric weight mapping (matches config/settings.py)
IMPORTANCE_WEIGHTS = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.3,
}


# ── Role Profile Builder ──────────────────────────────────────────────────────

def build_role_profiles(df: pd.DataFrame) -> dict:
    """
    Build a dictionary of role profiles from the processed dataframe.

    Structure
    ---------
    {
      "Data Scientist": [
        {"skill": "python",    "importance": "high",   "weight": 1.0, "category": "programming"},
        {"skill": "sql",       "importance": "high",   "weight": 1.0, "category": "database"},
        ...
      ],
      ...
    }

    Parameters
    ----------
    df : pd.DataFrame
        Processed role-skill dataframe

    Returns
    -------
    dict
        Role name → list of skill dicts
    """
    profiles = {}
    for role, group in df.groupby("role"):
        skills_list = []
        for _, row in group.iterrows():
            importance = row["importance"].strip().lower()
            skills_list.append({
                "skill":      row["skill"],
                "importance": importance,
                "weight":     IMPORTANCE_WEIGHTS.get(importance, 0.3),
                "category":   row.get("category", "general"),
            })
        profiles[role] = skills_list
    return profiles


# ── Skill Vocabulary ─────────────────────────────────────────────────────────

def build_skill_vocabulary(df: pd.DataFrame) -> list:
    """
    Return a sorted list of all unique skill names in the dataset.

    This vocabulary acts as the feature space (columns) for all
    binary and weighted matrices.
    """
    return sorted(df["skill"].unique().tolist())


# ── Binary Skill Matrix ───────────────────────────────────────────────────────

def build_binary_matrix(df: pd.DataFrame, vocabulary: list) -> pd.DataFrame:
    """
    Build a binary (0/1) skills matrix with shape (n_roles × n_skills).

    Cell[role, skill] = 1 if the role requires that skill, else 0.

    Used for quick presence/absence checks and as input to classifiers.
    """
    roles = sorted(df["role"].unique().tolist())
    matrix = pd.DataFrame(0, index=roles, columns=vocabulary)

    for _, row in df.iterrows():
        role  = row["role"]
        skill = row["skill"]
        if skill in vocabulary:
            matrix.loc[role, skill] = 1

    return matrix


# ── Weighted Skill Matrix ─────────────────────────────────────────────────────

def build_weighted_matrix(df: pd.DataFrame, vocabulary: list) -> pd.DataFrame:
    """
    Build a weighted skills matrix with shape (n_roles × n_skills).

    Cell[role, skill] = importance weight (1.0 / 0.6 / 0.3) or 0.

    Used for weighted cosine similarity — high-importance skills contribute
    more to the similarity score than low-importance ones.
    """
    roles = sorted(df["role"].unique().tolist())
    matrix = pd.DataFrame(0.0, index=roles, columns=vocabulary)

    for _, row in df.iterrows():
        role       = row["role"]
        skill      = row["skill"]
        importance = row["importance"].strip().lower()
        weight     = IMPORTANCE_WEIGHTS.get(importance, 0.3)
        if skill in vocabulary:
            matrix.loc[role, skill] = weight

    return matrix


# ── User Skill Vector ─────────────────────────────────────────────────────────

def build_user_vector(user_skills: list, vocabulary: list,
                      weighted: bool = False,
                      role_df: Optional[pd.DataFrame] = None,
                      target_role: Optional[str] = None) -> np.ndarray:
    """
    Convert a user's skill list into a numeric vector aligned to vocabulary.

    Parameters
    ----------
    user_skills : list of str
        Normalized skill names from the user
    vocabulary : list of str
        Full skill vocabulary (feature columns)
    weighted : bool
        If True and role_df + target_role provided, use role's importance
        weights for matched skills instead of binary 1.0
    role_df : pd.DataFrame, optional
        Processed dataframe — required if weighted=True
    target_role : str, optional
        Target job role name — required if weighted=True

    Returns
    -------
    np.ndarray
        1-D vector of length len(vocabulary)
    """
    vector = np.zeros(len(vocabulary), dtype=float)
    vocab_index = {skill: i for i, skill in enumerate(vocabulary)}

    if weighted and role_df is not None and target_role is not None:
        # Build a quick weight lookup for skills in the target role
        role_weights = {}
        role_rows = role_df[role_df["role"] == target_role]
        for _, row in role_rows.iterrows():
            imp = row["importance"].strip().lower()
            role_weights[row["skill"]] = IMPORTANCE_WEIGHTS.get(imp, 0.3)

        for skill in user_skills:
            if skill in vocab_index:
                # Use role weight if skill is relevant to the target role
                vector[vocab_index[skill]] = role_weights.get(skill, 1.0)
    else:
        for skill in user_skills:
            if skill in vocab_index:
                vector[vocab_index[skill]] = 1.0

    return vector


# ── Cosine Similarity Helper ──────────────────────────────────────────────────

def cosine_similarity_vectors(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    cos(θ) = (A · B) / (||A|| × ||B||)

    Returns 0.0 if either vector is all zeros (no division by zero).
    """
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ── Save / Load feature artifacts ────────────────────────────────────────────

def save_features(binary_matrix: pd.DataFrame,
                  weighted_matrix: pd.DataFrame,
                  vocabulary: list,
                  role_profiles: dict,
                  output_dir: str = PROCESSED_DIR) -> None:
    """Persist all feature artifacts to processed/ for reuse."""
    os.makedirs(output_dir, exist_ok=True)

    binary_matrix.to_csv(os.path.join(output_dir, "binary_matrix.csv"))
    weighted_matrix.to_csv(os.path.join(output_dir, "weighted_matrix.csv"))

    with open(os.path.join(output_dir, "vocabulary.json"), "w") as f:
        json.dump(vocabulary, f, indent=2)

    with open(os.path.join(output_dir, "role_profiles.json"), "w") as f:
        json.dump(role_profiles, f, indent=2)

    print(f"Feature artifacts saved to: {output_dir}")


def load_features(processed_dir: str = PROCESSED_DIR) -> tuple:
    """
    Load pre-computed feature artifacts.

    Returns
    -------
    tuple : (binary_matrix, weighted_matrix, vocabulary, role_profiles)
    """
    binary_matrix   = pd.read_csv(os.path.join(processed_dir, "binary_matrix.csv"), index_col=0)
    weighted_matrix = pd.read_csv(os.path.join(processed_dir, "weighted_matrix.csv"), index_col=0)

    with open(os.path.join(processed_dir, "vocabulary.json")) as f:
        vocabulary = json.load(f)

    with open(os.path.join(processed_dir, "role_profiles.json")) as f:
        role_profiles = json.load(f)

    return binary_matrix, weighted_matrix, vocabulary, role_profiles


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_CSV)

    print("Building role profiles...")
    profiles = build_role_profiles(df)
    print(f"  Roles built: {list(profiles.keys())}")

    print("Building skill vocabulary...")
    vocab = build_skill_vocabulary(df)
    print(f"  Vocabulary size: {len(vocab)} unique skills")

    print("Building binary skill matrix...")
    bin_mat = build_binary_matrix(df, vocab)
    print(f"  Binary matrix shape: {bin_mat.shape}")

    print("Building weighted skill matrix...")
    wgt_mat = build_weighted_matrix(df, vocab)
    print(f"  Weighted matrix shape: {wgt_mat.shape}")

    # Demo: user vector
    sample_user_skills = ["python", "sql", "pandas", "excel"]
    user_vec = build_user_vector(sample_user_skills, vocab)
    print(f"\nDemo user vector (non-zero positions): "
          f"{[vocab[i] for i in range(len(vocab)) if user_vec[i] > 0]}")

    # Demo: cosine similarity of user vs Data Scientist role
    ds_vec = wgt_mat.loc["Data Scientist"].values
    sim = cosine_similarity_vectors(user_vec, ds_vec)
    print(f"Cosine similarity (user vs Data Scientist): {sim:.4f}")

    print("\nSaving feature artifacts...")
    save_features(bin_mat, wgt_mat, vocab, profiles)
