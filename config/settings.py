# ─────────────────────────────────────────────
#  config/settings.py
#  Central configuration for the Skill Gap Analyzer
# ─────────────────────────────────────────────

import os

# ── Paths ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ── Dataset file names ───────────────────────
JOB_ROLES_CSV = os.path.join(RAW_DATA_DIR, "job_roles_skills.csv")
JOB_DESCRIPTIONS_JSON = os.path.join(RAW_DATA_DIR, "job_descriptions.json")
SKILL_ALIASES_JSON = os.path.join(RAW_DATA_DIR, "skill_aliases.json")
PROCESSED_ROLES_CSV = os.path.join(PROCESSED_DATA_DIR, "processed_roles.csv")

# ── Supported job roles ──────────────────────
SUPPORTED_ROLES = [
    "Data Scientist",
    "Data Analyst",
    "Machine Learning Engineer",
    "AI Engineer",
    "Python Developer",
    "Web Developer",
    "Cloud Engineer",
    "Cybersecurity Analyst",
    "Software Developer",
    "DevOps Engineer",
]

# ── Scoring weights ──────────────────────────
# Weight of each skill importance tier in the readiness score
WEIGHT_HIGH = 1.0       # Critical skills
WEIGHT_MEDIUM = 0.6     # Important but not critical
WEIGHT_LOW = 0.3        # Nice-to-have skills

# ── NLP settings ─────────────────────────────
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 2)
SIMILARITY_THRESHOLD = 0.70   # Cosine similarity threshold for skill match

# ── Partial match settings ───────────────────
PARTIAL_MATCH_THRESHOLD = 0.50   # Below this → missing; above → partial

# ── App display settings ─────────────────────
APP_TITLE = "Skill Gap Analyzer"
APP_SUBTITLE = "Understand the gap between your skills and your dream job."
