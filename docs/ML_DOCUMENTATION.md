# Machine Learning & NLP Documentation
## Skill Gap Analyzer — BCA ML Project

---

## 1. Dataset

| File | Format | Description |
|---|---|---|
| `data/raw/job_roles_skills.csv` | CSV | Structured role → skill mapping with importance tiers |
| `data/raw/job_descriptions.json` | JSON | One natural-language job description per role |
| `data/raw/skill_aliases.json` | JSON | Skill name normalization dictionary |
| `data/raw/learning_resources.json` | JSON | Curated learning resources per skill |

**Dataset size:** ~163 records · 10 job roles · ~80 unique skills

---

## 2. Features

### Structured features (from CSV)
- **Skill name** (normalized string)
- **Importance tier** (high / medium / low)
- **Importance weight** (1.0 / 0.6 / 0.3)
- **Skill category** (programming / ml / devops / etc.)

### NLP features (from job descriptions)
- **TF-IDF vectors** — n-gram (1,2) weighted term-frequency vectors
  - Each role's job description becomes a sparse vector of ~500 features
  - Terms unique to a role get higher weight (IDF component)
  - Sublinear TF scaling reduces impact of repeated terms

---

## 3. Preprocessing

Steps applied in [`src/preprocessing/data_cleaner.py`](../src/preprocessing/data_cleaner.py):

1. Lowercase all skill names
2. Strip leading/trailing whitespace
3. Collapse multiple spaces
4. Remove special characters (except `-` and `/`)
5. Apply alias dictionary substitution (`ml` → `machine learning`)
6. Validate importance tier values
7. Remove duplicate (role, skill) pairs
8. Sort for reproducibility

Text preprocessing for TF-IDF:
1. Lowercase job description text
2. Remove punctuation (keep `-` and `/`)
3. Tokenize as unigrams + bigrams
4. Remove English stopwords + domain stopwords
5. Apply minimum character filter (≥ 2 chars)

---

## 4. Model / Algorithm

### NLP Component — TF-IDF Vectorizer
```
Algorithm  : TF-IDF (Term Frequency–Inverse Document Frequency)
Library    : sklearn.feature_extraction.text.TfidfVectorizer
Parameters : max_features=500, ngram_range=(1,2), sublinear_tf=True
Input      : Raw job description text (one document per role)
Output     : Dense vector of shape (n_roles × 500)
```

**Why TF-IDF:**
- Captures vocabulary that is characteristic of each role
- Handles word-level and phrase-level (bigram) features
- More meaningful than raw bag-of-words for short domain texts
- Computationally lightweight — appropriate for a BCA-level project

### Similarity Component — Cosine Similarity
```
Formula : cos(θ) = (A · B) / (||A|| × ||B||)
Library : sklearn.metrics.pairwise.cosine_similarity
Input   : User skill text vector vs role description vectors
Output  : Similarity score ∈ [0, 1] per role
```

**Why cosine similarity (not Euclidean):**
- Invariant to vector magnitude — skill count differences don't penalize
- Standard metric for TF-IDF document matching
- Gives interpretable 0–1 scores

### Skill Name Matching — Character N-gram TF-IDF
```
Algorithm  : Character-level TF-IDF (char_wb, ngram=(2,4))
Purpose    : Partial/alias matching of skill names
Input      : Unmatched required skill names + unmatched user skill names
Output     : Cosine similarity matrix (required × user)
```

**Why character n-grams:**
- Handles slight misspellings: `scikit-learn` ↔ `scikitlearn`
- Catches abbreviations: `deep learning` ↔ `dl`
- More robust than exact string matching for short text

---

## 5. Training Process

The TF-IDF model is unsupervised — there are no labels, no train/test split.

```
Step 1 : Load 10 job descriptions (one per role)
Step 2 : fit_transform() builds vocabulary + computes TF-IDF weights
Step 3 : Fitted model is saved to models/tfidf_vectorizer.pkl
Step 4 : At inference time, transform() converts user skill text to a vector
Step 5 : cosine_similarity() compares user vector vs all role vectors
```

This is valid unsupervised NLP — the model learns what terms are significant
per role from the corpus, without any hand-labeled training data.

---

## 6. Prediction / Inference Process

```
User enters skills
        ↓
Normalize (lowercase, alias map)
        ↓
Pass 1: Exact string matching against role skill list
        ↓
Pass 2: Character TF-IDF cosine similarity for unmatched skills
        → Above 0.85 → Full match (NLP)
        → 0.45–0.85  → Partial match
        → Below 0.45 → Missing
        ↓
TF-IDF vectorize user skill text
        ↓
Cosine similarity vs all role description vectors → similarity scores
        ↓
Compute readiness score, gap score, match %
        ↓
Prioritize missing skills by importance weight
        ↓
Generate learning path
```

---

## 7. Scoring Logic

### Skill Match Percentage (SMP)
```
SMP = (n_matched / n_total_required) × 100
```
Binary: counts matched skills only, ignores partial.

### Weighted Readiness Score (RS)
```
earned = Σ weight(matched) + Σ (0.5 × weight(partial))
total  = Σ weight(all_required)
RS     = (earned / total) × 100
```
Where weights are: high=1.0, medium=0.6, low=0.3.

This gives a fairer score: a user who has all high-importance skills but
lacks nice-to-have skills scores much higher than a user missing core skills.

### Skill Gap Score (SGS)
```
SGS = 100 - RS
```

### Readiness Labels
| Score | Label |
|---|---|
| 0–29 | Beginner 🔰 |
| 30–49 | Developing 🌱 |
| 50–69 | Intermediate 📈 |
| 70–84 | Advanced ⭐ |
| 85–100 | Job-Ready 🚀 |

---

## 8. Evaluation

Since this is an unsupervised NLP system (no classification labels), traditional
accuracy metrics do not apply. Instead we evaluate:

| Criterion | How assessed |
|---|---|
| Skill coverage | All 10 roles have complete skill lists (manually verified) |
| Alias normalization | Unit tests verify `ml` → `machine learning` etc. |
| Match correctness | Integration test: Python/SQL/Pandas matched for Data Scientist |
| Score validity | Gap + Readiness = 100 (verified in tests) |
| Edge cases | Empty skills → score=0; all skills → score=100 (verified) |
| NLP quality | TF-IDF top terms manually inspected per role |

---

## 9. Limitations

| Limitation | Description |
|---|---|
| Small corpus | Only 10 job descriptions — TF-IDF generalizes less well than with 1000+ |
| Static dataset | Skills don't auto-update from live job boards |
| No proficiency levels | System assumes binary skill possession (have/don't have) |
| Importance manually assigned | High/medium/low tiers set by domain knowledge, not learned |
| No user history | No memory of previous analyses |
| English only | NLP processing assumes English input |

---

## 10. What Makes This Genuinely ML/NLP

This project uses **real, established NLP and ML techniques** — not fake or
simulated ML:

✅ **TF-IDF** — a standard NLP weighting scheme, implemented via sklearn  
✅ **Cosine Similarity** — standard vector similarity metric from linear algebra  
✅ **Character n-gram TF-IDF** — used for fuzzy skill name matching  
✅ **Unsupervised learning** — model learns from corpus without labels  
✅ **Feature engineering** — skills converted to numerical vectors  
✅ **Vectorization** — text data converted to numeric representation  

The system would not work identically if the job descriptions were swapped —
the TF-IDF weights change, which changes which skills are extracted and which
roles users are matched to. This is genuine data-driven behavior, not hardcoding.
