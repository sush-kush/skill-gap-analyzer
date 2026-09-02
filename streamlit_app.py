"""
streamlit_app.py  —  ROOT-LEVEL entry point for Streamlit Cloud
----------------------------------------------------------------
This single file embeds ALL pipeline logic so that Streamlit Cloud
can run it without any sys.path manipulation or subfolder imports.

Local run  :  streamlit run streamlit_app.py
Cloud run  :  set Main file path = streamlit_app.py
"""

import os
import re
import json
import pickle
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

# ── Root path (same dir as this file) ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DIR       = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")

ROLES_CSV        = os.path.join(RAW_DIR, "job_roles_skills.csv")
ALIASES_JSON     = os.path.join(RAW_DIR, "skill_aliases.json")
JD_JSON          = os.path.join(RAW_DIR, "job_descriptions.json")
RESOURCES_JSON   = os.path.join(RAW_DIR, "learning_resources.json")
PROCESSED_CSV    = os.path.join(PROCESSED_DIR, "processed_roles.csv")
VECTORIZER_PATH  = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

IMPORTANCE_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.3}
SUPPORTED_ROLES = [
    "Data Scientist", "Data Analyst", "Machine Learning Engineer",
    "AI Engineer", "Python Developer", "Web Developer",
    "Cloud Engineer", "Cybersecurity Analyst", "Software Developer",
    "DevOps Engineer",
]

logging.basicConfig(level=logging.WARNING)

# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — DATA CLEANING
# ═════════════════════════════════════════════════════════════════════════════

def load_aliases():
    with open(ALIASES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_skill(skill: str, aliases: dict) -> str:
    cleaned = skill.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\-/]", "", cleaned)
    return aliases.get(cleaned, cleaned)

def normalize_user_skills(raw: list, aliases: dict) -> list:
    seen, result = set(), []
    for s in raw:
        n = normalize_skill(s, aliases)
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    return result

def clean_and_load() -> pd.DataFrame:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if os.path.exists(PROCESSED_CSV):
        return pd.read_csv(PROCESSED_CSV)
    aliases = load_aliases()
    df = pd.read_csv(ROLES_CSV)
    df = df.dropna(subset=["role", "skill"])
    df["role"]       = df["role"].str.strip().str.title()
    df["skill"]      = df["skill"].apply(lambda s: normalize_skill(s, aliases))
    df["importance"] = df["importance"].str.strip().str.lower()
    df.loc[~df["importance"].isin({"high","medium","low"}), "importance"] = "medium"
    df["category"]   = df["category"].str.strip().str.lower().fillna("general")
    df = df.drop_duplicates(subset=["role", "skill"])
    df = df.sort_values(["role","importance","skill"]).reset_index(drop=True)
    df.to_csv(PROCESSED_CSV, index=False)
    return df

# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def build_role_profiles(df: pd.DataFrame) -> dict:
    profiles = {}
    for role, grp in df.groupby("role"):
        profiles[role] = [
            {
                "skill":      r["skill"],
                "importance": r["importance"],
                "weight":     IMPORTANCE_WEIGHTS.get(r["importance"], 0.3),
                "category":   r.get("category", "general"),
            }
            for _, r in grp.iterrows()
        ]
    return profiles

# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — NLP / TF-IDF
# ═════════════════════════════════════════════════════════════════════════════

def _preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\-/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

class SkillExtractor:
    def __init__(self):
        self.vectorizer   = TfidfVectorizer(
            max_features=500, ngram_range=(1, 2),
            stop_words="english", preprocessor=_preprocess_text,
            token_pattern=r"[a-zA-Z][a-zA-Z0-9\-/]{1,}",
            sublinear_tf=True,
        )
        self.role_names   = []
        self.tfidf_matrix = None
        self.vocabulary_  = []
        self._fitted      = False

    def fit(self, jd: dict):
        self.role_names   = list(jd.keys())
        corpus            = [jd[r] for r in self.role_names]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()
        self.vocabulary_  = self.vectorizer.get_feature_names_out().tolist()
        self._fitted      = True
        return self

    def user_role_similarity(self, user_skills: list) -> dict:
        if not self._fitted:
            return {r: 0.0 for r in self.role_names}
        user_vec = self.vectorizer.transform([" ".join(user_skills)]).toarray()
        sims     = sk_cosine_similarity(user_vec, self.tfidf_matrix)[0]
        return {r: float(round(sims[i], 4)) for i, r in enumerate(self.role_names)}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

def get_extractor() -> SkillExtractor:
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(VECTORIZER_PATH):
        try:
            return SkillExtractor.load(VECTORIZER_PATH)
        except Exception:
            pass
    with open(JD_JSON, "r", encoding="utf-8") as f:
        jd = json.load(f)
    ext = SkillExtractor()
    ext.fit(jd)
    ext.save(VECTORIZER_PATH)
    return ext

# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — SKILL MATCHING
# ═════════════════════════════════════════════════════════════════════════════

def match_skills(user_skills: list, role_reqs: list) -> dict:
    """
    Returns dict with keys: matched, partial, missing
    Each value is a list of requirement dicts augmented with status info.
    """
    user_set       = set(user_skills)
    matched, unmatched_reqs, unmatched_user = [], [], list(user_skills)

    # Pass 1 — exact
    for req in role_reqs:
        if req["skill"] in user_set:
            matched.append({**req, "status": "matched", "how": "exact", "user_skill": req["skill"], "sim": 1.0})
            if req["skill"] in unmatched_user:
                unmatched_user.remove(req["skill"])
        else:
            unmatched_reqs.append(req)

    partial, missing = [], []

    # Pass 2 — character n-gram TF-IDF similarity
    if unmatched_reqs and unmatched_user:
        all_terms = unmatched_user + [r["skill"] for r in unmatched_reqs]
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            mat = vec.fit_transform(all_terms).toarray()
            n   = len(unmatched_user)
            sim_mat = sk_cosine_similarity(mat[n:], mat[:n])
            used = set()
            for r_idx, req in enumerate(unmatched_reqs):
                sims    = sim_mat[r_idx]
                best_i  = int(np.argmax(sims))
                best_s  = float(sims[best_i])
                if best_s >= 0.85 and best_i not in used:
                    matched.append({**req, "status": "matched", "how": "nlp",
                                    "user_skill": unmatched_user[best_i], "sim": round(best_s, 3)})
                    used.add(best_i)
                elif best_s >= 0.45 and best_i not in used:
                    partial.append({**req, "status": "partial", "how": "nlp",
                                    "user_skill": unmatched_user[best_i], "sim": round(best_s, 3)})
                    used.add(best_i)
                else:
                    missing.append({**req, "status": "missing"})
        except Exception:
            missing.extend([{**r, "status": "missing"} for r in unmatched_reqs])
    else:
        missing.extend([{**r, "status": "missing"} for r in unmatched_reqs])

    return {"matched": matched, "partial": partial, "missing": missing}

# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — SCORING
# ═════════════════════════════════════════════════════════════════════════════

def compute_scores(match: dict) -> dict:
    matched = match["matched"]
    partial = match["partial"]
    missing = match["missing"]
    all_req = matched + partial + missing
    total   = len(all_req)

    skill_match_pct = round(len(matched) / total * 100, 1) if total else 0.0

    total_w  = sum(r["weight"] for r in all_req)
    earned_w = sum(r["weight"] for r in matched) + sum(r["weight"] * 0.5 for r in partial)
    readiness = round(earned_w / total_w * 100, 1) if total_w else 0.0
    gap       = round(100.0 - readiness, 1)

    if readiness >= 85:   label = "Job-Ready"
    elif readiness >= 70: label = "Advanced"
    elif readiness >= 50: label = "Intermediate"
    elif readiness >= 30: label = "Developing"
    else:                 label = "Beginner"

    # Category breakdown
    from collections import defaultdict
    cat_data = defaultdict(lambda: {"earned": 0.0, "total": 0.0, "matched": 0, "partial": 0, "missing": 0})
    for r in matched:
        cat_data[r["category"]]["earned"] += r["weight"]
        cat_data[r["category"]]["total"]  += r["weight"]
        cat_data[r["category"]]["matched"] += 1
    for r in partial:
        cat_data[r["category"]]["earned"] += r["weight"] * 0.5
        cat_data[r["category"]]["total"]  += r["weight"]
        cat_data[r["category"]]["partial"] += 1
    for r in missing:
        cat_data[r["category"]]["total"]  += r["weight"]
        cat_data[r["category"]]["missing"] += 1
    cat_breakdown = {
        c: {
            "pct": round(d["earned"] / d["total"] * 100, 1) if d["total"] else 0.0,
            "matched": d["matched"], "partial": d["partial"], "missing": d["missing"],
        }
        for c, d in cat_data.items()
    }

    return {
        "total": total,
        "n_matched": len(matched), "n_partial": len(partial), "n_missing": len(missing),
        "skill_match_pct": skill_match_pct,
        "readiness": readiness, "gap": gap, "label": label,
        "cat_breakdown": cat_breakdown,
        "high_missing":   [r["skill"] for r in missing if r["importance"] == "high"],
        "medium_missing": [r["skill"] for r in missing if r["importance"] == "medium"],
        "low_missing":    [r["skill"] for r in missing if r["importance"] == "low"],
    }

# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 6 — RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════

def load_resources() -> dict:
    if os.path.exists(RESOURCES_JSON):
        with open(RESOURCES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def build_recommendations(match: dict, score: dict, role: str, resources: dict) -> list:
    recs, rank = [], 1
    priority_order = [
        ("high",   [r for r in match["missing"] if r["importance"] == "high"],   "CRITICAL skill for {}. Highest impact on readiness."),
        ("deepen", match["partial"],                                               "You partially know '{}'. Deepen it for {}."),
        ("medium", [r for r in match["missing"] if r["importance"] == "medium"],  "Important skill for {}. Learn after high-priority gaps."),
        ("low",    [r for r in match["missing"] if r["importance"] == "low"],     "Nice-to-have for {}."),
    ]
    for priority, items, reason_tpl in priority_order:
        for item in items:
            skill = item["skill"]
            if priority == "deepen":
                reason = reason_tpl.format(item.get("user_skill", skill), role)
            else:
                reason = reason_tpl.format(role)
            res_data  = resources.get(skill, {})
            res_list  = res_data.get("resources", [])
            if not res_list:
                res_list = [{"title": "Search: '{}' tutorial".format(skill),
                             "url": "https://www.google.com/search?q={}+tutorial".format(skill.replace(" ", "+")),
                             "type": "search", "duration": "varies"}]
            recs.append({
                "rank": rank, "skill": skill, "priority": priority,
                "reason": reason, "description": res_data.get("description", ""),
                "resources": res_list,
            })
            rank += 1
    return recs

def build_summary(score: dict, role: str, n_recs: int) -> str:
    r = score["readiness"]
    if r >= 85:   return "You are nearly job-ready for {}! Focus on {} remaining skills.".format(role, n_recs)
    elif r >= 70: return "Strong profile for {}. Filling {} skill gaps will make you highly competitive.".format(role, n_recs)
    elif r >= 50: return "Good foundation for {}. You need {} more skills — start with high-priority ones.".format(role, n_recs)
    elif r >= 30: return "On the right track for {}. Commit to the {}-skill learning path.".format(role, n_recs)
    else:         return "Starting your journey toward {}. Begin with the {} high-priority foundational skills.".format(role, len(score["high_missing"]))

# ═════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.score-card{background:#f7f8fa;border-radius:10px;padding:1.2rem 1.5rem;
    text-align:center;border:1px solid #e5e7eb;}
.score-value{font-size:2.2rem;font-weight:700;}
.score-label{font-size:0.82rem;color:#57606a;text-transform:uppercase;letter-spacing:.05em;}
.skill-chip{display:inline-block;padding:.25rem .75rem;border-radius:20px;
    font-size:.82rem;margin:.18rem;font-weight:500;}
.matched-chip{background:#dcfce7;color:#166534;}
.partial-chip{background:#fef9c3;color:#854d0e;}
.missing-chip{background:#fee2e2;color:#991b1b;}
.rec-card{background:#fff;border-left:4px solid;border-radius:6px;
    padding:.85rem 1.1rem;margin-bottom:.7rem;box-shadow:0 1px 3px rgba(0,0,0,.07);}
</style>
""", unsafe_allow_html=True)

# ── Cached pipeline load ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Setting up pipeline...")
def load_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    df        = clean_and_load()
    profiles  = build_role_profiles(df)
    extractor = get_extractor()
    resources = load_resources()
    aliases   = load_aliases()
    return df, profiles, extractor, resources, aliases

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Skill Gap Analyzer")
    st.markdown("---")
    st.markdown("""
**How it works:**
1. Enter your current skills
2. Select your target job role
3. Click **Analyze**
4. View gap analysis & learn
""")
    st.markdown("---")
    st.markdown("**Tech Stack:**")
    st.markdown("Python · Pandas · Scikit-learn · TF-IDF · Streamlit")
    st.markdown("---")
    st.caption("BCA ML Project · Skill Gap Analyzer v1.0")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_analyze, tab_eda = st.tabs(["🔍 Analyze Skills", "📊 Explore Dataset"])

with tab_analyze:
    st.markdown("# 🎯 Skill Gap Analyzer")
    st.markdown("##### Understand the gap between your skills and your dream job.")
    st.markdown("---")

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.subheader("📝 Your Current Skills")
        skills_input = st.text_area(
            "Enter your skills (comma-separated or one per line):",
            placeholder="e.g. Python, SQL, Pandas, Excel",
            height=155,
        )
        st.caption("Tip: Variations like 'ML', 'sklearn', 'dl' are handled automatically.")
    with col2:
        st.subheader("🎯 Target Job Role")
        target_role = st.selectbox("Select a role:", SUPPORTED_ROLES)
        st.markdown("")
        analyze = st.button("🔍 Analyze My Skills", type="primary", use_container_width=True)

    if analyze:
        # Parse skills
        raw = []
        for line in skills_input.splitlines():
            raw.extend([s.strip() for s in line.split(",") if s.strip()])

        if not raw:
            st.warning("Please enter at least one skill before analyzing.")
            st.stop()

        with st.spinner("Analyzing your skill profile..."):
            df, profiles, extractor, resources, aliases = load_pipeline()
            user_skills   = normalize_user_skills(raw, aliases)
            role_reqs     = profiles.get(target_role, [])
            match         = match_skills(user_skills, role_reqs)
            score         = compute_scores(match)
            nlp_sims      = extractor.user_role_similarity(user_skills)
            target_sim    = nlp_sims.get(target_role, 0.0)
            recs          = build_recommendations(match, score, target_role, resources)
            summary       = build_summary(score, target_role, len(recs))

        st.success("Analysis complete for **{}**".format(target_role))
        st.markdown("---")

        # ── Score cards ───────────────────────────────────────────────────────
        st.markdown("### 📊 Your Skill Gap Scores")
        rc = "#22c55e" if score["readiness"] >= 70 else "#f59e0b" if score["readiness"] >= 40 else "#ef4444"
        gc = "#ef4444" if score["gap"] >= 60 else "#f59e0b" if score["gap"] >= 30 else "#22c55e"
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown('<div class="score-card"><div class="score-value" style="color:{}">{:.1f}%</div><div class="score-label">Readiness Score</div></div>'.format(rc, score["readiness"]), unsafe_allow_html=True)
        c2.markdown('<div class="score-card"><div class="score-value" style="color:{}">{:.1f}%</div><div class="score-label">Skill Gap Score</div></div>'.format(gc, score["gap"]), unsafe_allow_html=True)
        c3.markdown('<div class="score-card"><div class="score-value" style="color:#3b82d4">{:.1f}%</div><div class="score-label">Skills Matched</div></div>'.format(score["skill_match_pct"]), unsafe_allow_html=True)
        c4.markdown('<div class="score-card"><div class="score-value" style="color:#7c5cd8;font-size:1.4rem">{}</div><div class="score-label">Your Level</div></div>'.format(score["label"]), unsafe_allow_html=True)
        st.markdown("")

        # ── Gauge ─────────────────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score["readiness"],
            title={"text": "Job Readiness — {}".format(target_role), "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#3b82d4"},
                "steps": [
                    {"range": [0,  30], "color": "#fee2e2"},
                    {"range": [30, 50], "color": "#fef9c3"},
                    {"range": [50, 70], "color": "#fef3c7"},
                    {"range": [70, 85], "color": "#d1fae5"},
                    {"range": [85,100], "color": "#bbf7d0"},
                ],
                "threshold": {"line": {"color": "green", "width": 3}, "thickness": 0.75, "value": 85},
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("---")

        # ── Skill chips ───────────────────────────────────────────────────────
        st.markdown("### 🔍 Skill Breakdown")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Matched ({})**".format(score["n_matched"]))
            if match["matched"]:
                st.markdown(" ".join('<span class="skill-chip matched-chip">{}</span>'.format(s["skill"]) for s in match["matched"]), unsafe_allow_html=True)
            else:
                st.caption("None matched")
        with sc2:
            st.markdown("**Partial ({})**".format(score["n_partial"]))
            if match["partial"]:
                st.markdown(" ".join('<span class="skill-chip partial-chip">{}</span>'.format(s["skill"]) for s in match["partial"]), unsafe_allow_html=True)
            else:
                st.caption("None")
        with sc3:
            st.markdown("**Missing ({})**".format(score["n_missing"]))
            if match["missing"]:
                imp_order = ["high", "medium", "low"]
                sorted_m  = sorted(match["missing"], key=lambda x: imp_order.index(x["importance"]))
                st.markdown(" ".join('<span class="skill-chip missing-chip">{} [{}]</span>'.format(s["skill"], s["importance"][0].upper()) for s in sorted_m), unsafe_allow_html=True)
            else:
                st.success("No missing skills!")
        st.markdown("---")

        # ── Charts ────────────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_pie = go.Figure(go.Pie(
                labels=["Matched", "Partial", "Missing"],
                values=[score["n_matched"], score["n_partial"], score["n_missing"]],
                hole=0.55,
                marker_colors=["#22c55e", "#f59e0b", "#ef4444"],
                textinfo="label+percent",
            ))
            fig_pie.update_layout(title="Skill Coverage", height=300, margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            if score["cat_breakdown"]:
                cats  = list(score["cat_breakdown"].keys())
                pcts  = [score["cat_breakdown"][c]["pct"] for c in cats]
                ccols = ["#22c55e" if p >= 70 else "#f59e0b" if p >= 40 else "#ef4444" for p in pcts]
                fig_bar = go.Figure(go.Bar(
                    x=pcts, y=cats, orientation="h",
                    marker_color=ccols,
                    text=["{}%".format(p) for p in pcts], textposition="outside",
                ))
                fig_bar.update_layout(title="Readiness by Category",
                                      xaxis=dict(range=[0,115]), height=300,
                                      margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("---")

        # ── NLP similarity table ──────────────────────────────────────────────
        st.markdown("### 🤖 NLP Similarity Scores (TF-IDF Cosine Similarity)")
        st.caption("How closely your skill profile matches each role's job description text.")
        sim_df = pd.DataFrame(
            sorted(nlp_sims.items(), key=lambda x: -x[1]),
            columns=["Role", "NLP Similarity"]
        )
        sim_df["NLP Similarity"] = sim_df["NLP Similarity"].apply(lambda x: "{:.1%}".format(x))
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
        st.markdown("---")

        # ── Learning path ─────────────────────────────────────────────────────
        st.markdown("### 🗺️ Personalized Learning Path")
        st.info(summary)

        if not recs:
            st.success("You already have all required skills for this role!")
        else:
            p_headers = {
                "high":   "🔴 HIGH PRIORITY — Critical Missing Skills",
                "deepen": "🟡 DEEPEN — Strengthen Partial Skills",
                "medium": "🟠 MEDIUM PRIORITY — Important Skills",
                "low":    "🟢 LOW PRIORITY — Nice-to-Have",
            }
            p_colors = {"high": "#ef4444", "deepen": "#f59e0b", "medium": "#f97316", "low": "#22c55e"}
            for p in ["high", "deepen", "medium", "low"]:
                grp = [r for r in recs if r["priority"] == p]
                if not grp:
                    continue
                st.markdown("#### {}".format(p_headers[p]))
                for rec in grp:
                    bc = p_colors[p]
                    desc_html = "<br><small>{}</small>".format(rec["description"]) if rec["description"] else ""
                    st.markdown(
                        '<div class="rec-card" style="border-left-color:{}">'
                        '<b>#{} {}</b>{}'
                        '<br><small style="color:#57606a">{}</small></div>'.format(
                            bc, rec["rank"], rec["skill"].title(), desc_html, rec["reason"]
                        ),
                        unsafe_allow_html=True,
                    )
                    for res in rec["resources"][:2]:
                        st.markdown("&nbsp;&nbsp;&nbsp;→ **[{}]({})** *({} · {})*".format(
                            res["title"], res["url"], res.get("type","").upper(), res.get("duration","")
                        ))
                st.markdown("---")

        # ── Alternative roles ─────────────────────────────────────────────────
        alt = sorted([(r, s) for r, s in nlp_sims.items() if r != target_role], key=lambda x: -x[1])[:3]
        if alt:
            st.markdown("#### 💡 Alternative Roles You Might Be Closer To")
            ac = st.columns(3)
            for i, (r, s) in enumerate(alt):
                ac[i].metric(r, "{:.0%}".format(s), "NLP similarity")

    else:
        st.markdown("---")
        st.info("Enter your skills and select a target role above, then click **Analyze My Skills**.")
        with st.expander("📌 See an example"):
            st.markdown("""
**Skills:** Python, SQL, Pandas, Excel

**Target:** Data Scientist

**Output you'll see:**
- Matched: python, sql, pandas
- Missing: machine learning, statistics, numpy, scikit-learn...
- Readiness: ~21% (Beginner)
- Learning path: 17 skills with course links
""")

with tab_eda:
    st.markdown("### 📊 Dataset Explorer")
    df_eda, _, _, _, _ = load_pipeline()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records",  len(df_eda))
    c2.metric("Unique Roles",   df_eda["role"].nunique())
    c3.metric("Unique Skills",  df_eda["skill"].nunique())

    st.markdown("#### Top 20 Most In-Demand Skills")
    top_s = df_eda["skill"].value_counts().head(20)
    fig_t = px.bar(top_s, orientation="h", labels={"value": "# Roles", "index": "Skill"},
                   color=top_s.values, color_continuous_scale="Blues")
    fig_t.update_layout(height=450, showlegend=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("#### Skills per Role")
    rc2 = df_eda.groupby("role")["skill"].count().sort_values()
    fig_r = px.bar(rc2, orientation="h", labels={"value": "# Skills", "index": "Role"},
                   color=rc2.values, color_continuous_scale="Purples")
    fig_r.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("#### Browse Dataset")
    rf = st.selectbox("Filter by role:", ["All"] + SUPPORTED_ROLES, key="eda_r")
    fd = df_eda if rf == "All" else df_eda[df_eda["role"] == rf]
    st.dataframe(fd[["role", "skill", "importance", "category"]], use_container_width=True, height=300)
