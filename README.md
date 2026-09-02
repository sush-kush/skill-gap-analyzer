# Skill Gap Analyzer

> An ML/NLP-powered tool for students, fresh graduates, and job seekers to understand the gap between their current skills and a target job role.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Usage](#usage)
6. [ML/NLP Approach](#mlnlp-approach)
7. [Scoring Logic](#scoring-logic)
8. [Supported Job Roles](#supported-job-roles)
9. [Dataset](#dataset)
10. [Extending the Project](#extending-the-project)
11. [Limitations](#limitations)

---

## Project Overview

The **Skill Gap Analyzer** takes a user's current skills and a target job role, then:

- Identifies **matched**, **missing**, and **partially matched** skills
- Calculates a **Skill Gap Score** and **Job Readiness Score**
- Provides a **prioritized learning path** with recommended resources
- Uses **NLP (TF-IDF + Cosine Similarity)** to extract and match skills from job descriptions

---

## Features

| Feature | Description |
|---|---|
| Skill Matching | Exact + fuzzy/semantic matching of user skills vs role requirements |
| NLP Extraction | Skills extracted from raw job descriptions using TF-IDF |
| Gap Scoring | Mathematically defined readiness and gap scores |
| Recommendations | Priority-ordered learning path for missing skills |
| Visual Dashboard | Streamlit UI with charts and skill breakdown |
| Extensible Data | Add new roles/skills via CSV/JSON — no code changes needed |

---

## Project Structure

```
skill_gap_analyzer/
├── config/
│   └── settings.py          # Central configuration
├── data/
│   ├── raw/
│   │   ├── job_roles_skills.csv    # Structured role-skill dataset
│   │   ├── job_descriptions.json   # Raw job description texts
│   │   └── skill_aliases.json      # Skill name normalization map
│   └── processed/
│       └── processed_roles.csv     # Cleaned, normalized dataset
├── src/
│   ├── preprocessing/
│   │   └── data_cleaner.py         # Skill normalization & cleaning
│   ├── eda/
│   │   └── explorer.py             # EDA charts and statistics
│   ├── features/
│   │   └── feature_engineering.py  # Skill vectors, role profiles
│   ├── nlp/
│   │   └── skill_extractor.py      # TF-IDF skill extraction from JDs
│   ├── matching/
│   │   └── skill_matcher.py        # Matched/missing/partial logic
│   ├── scoring/
│   │   └── score_engine.py         # Gap score & readiness score
│   └── recommendations/
│       └── recommender.py          # Personalized learning path
├── app/
│   └── streamlit_app.py            # Streamlit UI
├── models/                          # Saved TF-IDF vectorizers
├── reports/                         # EDA outputs, charts
├── logs/                            # App logs
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/skill-gap-analyzer.git
cd skill-gap-analyzer

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 5. Run the app
streamlit run app/streamlit_app.py
```

---

## Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Enter your current skills (comma-separated)
3. Select your target job role
4. Click **Analyze**
5. View your:
   - Skill match breakdown
   - Missing & priority skills
   - Readiness score
   - Recommended learning path

---

## ML/NLP Approach

| Component | Technique | Purpose |
|---|---|---|
| Skill Extraction | TF-IDF + keyword filtering | Extract skills from raw job descriptions |
| Skill Matching | Cosine Similarity (TF-IDF vectors) | Match user skills to role requirements |
| Normalization | Custom alias map + lowercasing | Handle `ML` vs `Machine Learning` etc. |
| Prioritization | Frequency + importance weights | Rank missing skills by criticality |

---

## Scoring Logic

### Skill Match Percentage
```
match_pct = (matched_skills / total_required_skills) × 100
```

### Readiness Score (weighted)
```
readiness = Σ(matched_weight) / Σ(total_weight) × 100

where weight = 1.0 (critical) | 0.6 (important) | 0.3 (nice-to-have)
```

### Skill Gap Score
```
gap_score = 100 - readiness_score
```

---

## Supported Job Roles

- Data Scientist
- Data Analyst
- Machine Learning Engineer
- AI Engineer
- Python Developer
- Web Developer
- Cloud Engineer
- Cybersecurity Analyst
- Software Developer
- DevOps Engineer

---

## Dataset

- **`job_roles_skills.csv`** — Structured mapping of roles to required skills with importance tiers
- **`job_descriptions.json`** — Raw job description paragraphs for NLP extraction
- **`skill_aliases.json`** — Normalization dictionary (`"ml": "machine learning"`, etc.)

All datasets are in `data/raw/` and can be updated without changing application code.

---

## Extending the Project

To add a new job role:
1. Add rows to `data/raw/job_roles_skills.csv`
2. Add a job description to `data/raw/job_descriptions.json`
3. Add any new skill aliases to `data/raw/skill_aliases.json`
4. Re-run preprocessing: `python src/preprocessing/data_cleaner.py`

---

## Limitations

- Skill importance weights are manually assigned (future: learn from job posting data)
- Partial match detection is similarity-based and may misclassify niche skills
- Job descriptions are curated, not scraped live
- No user authentication or history tracking in v1

---

*Built as a BCA ML project. Suitable for academic submission and GitHub portfolio.*
