"""
src/preprocessing/data_cleaner.py

Stage 3 — Data Cleaning & Preprocessing
----------------------------------------
Responsibilities:
  - Load raw job_roles_skills.csv
  - Normalize skill names (lowercase, strip whitespace, apply alias map)
  - Remove duplicate role-skill pairs
  - Validate importance tier values
  - Save cleaned dataset to data/processed/processed_roles.csv

Why this matters:
  Downstream NLP and matching modules depend on consistent skill strings.
  Without normalization, "Machine Learning" and "machine learning" would be
  treated as different skills, breaking similarity calculations.
"""

import os
import json
import re
import logging
import pandas as pd

# ── Resolve paths relative to this file ─────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

RAW_CSV = os.path.join(_ROOT, "data", "raw", "job_roles_skills.csv")
ALIASES_JSON = os.path.join(_ROOT, "data", "raw", "skill_aliases.json")
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
PROCESSED_CSV = os.path.join(PROCESSED_DIR, "processed_roles.csv")

VALID_IMPORTANCE = {"high", "medium", "low"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_aliases(aliases_path: str) -> dict:
    """Load the skill alias map from JSON."""
    with open(aliases_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_skill(skill: str, aliases: dict) -> str:
    """
    Normalize a single skill string:
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Collapse multiple internal spaces
      4. Apply alias substitution

    Parameters
    ----------
    skill : str
        Raw skill string, e.g. "Machine Learning" or "ml"
    aliases : dict
        Mapping of raw variants → canonical names

    Returns
    -------
    str
        Normalized canonical skill name
    """
    cleaned = skill.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)      # collapse multiple spaces
    cleaned = re.sub(r"[^\w\s\-/]", "", cleaned)  # remove special chars except - / 
    return aliases.get(cleaned, cleaned)          # alias lookup


def normalize_role(role: str) -> str:
    """Title-case and strip a job role string."""
    return role.strip().title()


# ── Main cleaning pipeline ───────────────────────────────────────────────────

def clean_dataset(raw_csv: str = RAW_CSV,
                  aliases_path: str = ALIASES_JSON,
                  output_csv: str = PROCESSED_CSV) -> pd.DataFrame:
    """
    Full cleaning pipeline for the role-skill dataset.

    Steps
    -----
    1. Load raw CSV
    2. Drop rows with null role or skill
    3. Normalize role and skill names
    4. Apply skill alias map
    5. Validate importance column
    6. Remove duplicate (role, skill) pairs
    7. Save to processed CSV

    Parameters
    ----------
    raw_csv : str
        Path to raw job_roles_skills.csv
    aliases_path : str
        Path to skill_aliases.json
    output_csv : str
        Destination path for cleaned CSV

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    logger.info("Loading raw dataset from: %s", raw_csv)
    df = pd.read_csv(raw_csv)
    logger.info("Raw dataset shape: %s", df.shape)

    # ── 1. Drop rows missing critical columns ────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["role", "skill"])
    dropped_nulls = before - len(df)
    if dropped_nulls:
        logger.warning("Dropped %d rows with null role/skill values.", dropped_nulls)

    # ── 2. Load alias dictionary ─────────────────────────────────────────────
    aliases = load_aliases(aliases_path)
    logger.info("Loaded %d skill aliases.", len(aliases))

    # ── 3. Normalize role and skill ──────────────────────────────────────────
    df["role"] = df["role"].apply(normalize_role)
    df["skill_raw"] = df["skill"].copy()          # keep original for reference
    df["skill"] = df["skill"].apply(lambda s: normalize_skill(s, aliases))

    # ── 4. Validate importance tier ──────────────────────────────────────────
    df["importance"] = df["importance"].str.strip().str.lower()
    invalid_rows = df[~df["importance"].isin(VALID_IMPORTANCE)]
    if not invalid_rows.empty:
        logger.warning(
            "Found %d rows with invalid importance values. Setting to 'medium'.",
            len(invalid_rows)
        )
        df.loc[~df["importance"].isin(VALID_IMPORTANCE), "importance"] = "medium"

    # ── 5. Normalize category ────────────────────────────────────────────────
    df["category"] = df["category"].str.strip().str.lower().fillna("general")

    # ── 6. Remove duplicate (role, skill) pairs ──────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["role", "skill"])
    deduped = before_dedup - len(df)
    if deduped:
        logger.info("Removed %d duplicate (role, skill) pairs.", deduped)

    # ── 7. Sort for readability ──────────────────────────────────────────────
    df = df.sort_values(["role", "importance", "skill"]).reset_index(drop=True)

    # ── 8. Save processed CSV ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Processed dataset saved to: %s", output_csv)
    logger.info("Final dataset shape: %s", df.shape)

    return df


# ── Utility: normalize a list of user-input skills ──────────────────────────

def normalize_user_skills(user_skills: list, aliases=None) -> list:
    """
    Normalize a list of raw skill strings entered by the user.

    Parameters
    ----------
    user_skills : list of str
        Skills as typed by the user
    aliases : dict, optional
        Alias map; loaded from file if not provided

    Returns
    -------
    list of str
        Normalized, deduplicated skill names
    """
    if aliases is None:
        aliases = load_aliases(ALIASES_JSON)

    normalized = [normalize_skill(s, aliases) for s in user_skills if s.strip()]
    return list(dict.fromkeys(normalized))   # deduplicate, preserve order


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cleaned_df = clean_dataset()
    print("\n=== Sample of Cleaned Dataset ===")
    print(cleaned_df.head(20).to_string(index=False))
    print(f"\nTotal records : {len(cleaned_df)}")
    print(f"Unique roles  : {cleaned_df['role'].nunique()}")
    print(f"Unique skills : {cleaned_df['skill'].nunique()}")
