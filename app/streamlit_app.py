"""
app/streamlit_app.py

Stage 10 — Streamlit UI
------------------------
Full interactive web application for the Skill Gap Analyzer.

Pages / sections:
  1. Home / Input       — skill entry + role selection
  2. Analysis Dashboard — gap breakdown, scores, charts
  3. Learning Path      — prioritized recommendations
  4. EDA Explorer       — dataset exploration (optional tab)
"""

import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Path setup (works locally AND on Streamlit Cloud) ────────────────────────
# __file__ may be  .../app/streamlit_app.py  OR  .../streamlit_app.py
# depending on how Streamlit Cloud resolves the entry point.
# We always derive _ROOT as the directory that contains the "src" folder.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk up until we find the directory that has a "src" sub-folder
_ROOT = _APP_DIR
for _ in range(4):
    if os.path.isdir(os.path.join(_ROOT, "src")):
        break
    _ROOT = os.path.dirname(_ROOT)

sys.path.insert(0, os.path.join(_ROOT, "src"))

from preprocessing.data_cleaner import clean_dataset, normalize_user_skills, load_aliases
from features.feature_engineering import build_role_profiles
from matching.skill_matcher import SkillMatcher
from scoring.score_engine import compute_scores, ScoreResult
from recommendations.recommender import SkillRecommender, LearningPath
from nlp.skill_extractor import get_extractor

PROCESSED_CSV   = os.path.join(_ROOT, "data", "processed", "processed_roles.csv")
ALIASES_JSON    = os.path.join(_ROOT, "data", "raw", "skill_aliases.json")

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

PRIORITY_COLORS = {
    "high":   "#ef4444",
    "deepen": "#f59e0b",
    "medium": "#f97316",
    "low":    "#22c55e",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f2328;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #57606a;
        margin-bottom: 1.5rem;
    }
    .score-card {
        background: #f7f8fa;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .score-value {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .score-label {
        font-size: 0.85rem;
        color: #57606a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .skill-chip {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    .matched-chip  { background: #dcfce7; color: #166534; }
    .partial-chip  { background: #fef9c3; color: #854d0e; }
    .missing-chip  { background: #fee2e2; color: #991b1b; }
    .rec-card {
        background: #ffffff;
        border-left: 4px solid;
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    }
    .section-divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state & caching ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading dataset and models...")
def load_pipeline():
    """
    Load and cache the entire pipeline (runs once per session).
    On Streamlit Cloud the processed/ and models/ dirs don't exist yet,
    so we run the full setup pipeline on first load.
    """
    processed_dir = os.path.join(_ROOT, "data", "processed")
    models_dir    = os.path.join(_ROOT, "models")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Run cleaning if processed CSV is missing
    if not os.path.exists(PROCESSED_CSV):
        clean_dataset(
            raw_csv=os.path.join(_ROOT, "data", "raw", "job_roles_skills.csv"),
            aliases_path=os.path.join(_ROOT, "data", "raw", "skill_aliases.json"),
            output_csv=PROCESSED_CSV,
        )

    df       = pd.read_csv(PROCESSED_CSV)
    profiles = build_role_profiles(df)
    matcher  = SkillMatcher(role_profiles=profiles)
    recommender = SkillRecommender(
        resources_path=os.path.join(_ROOT, "data", "raw", "learning_resources.json")
    )
    extractor = get_extractor(
        jd_path=os.path.join(_ROOT, "data", "raw", "job_descriptions.json"),
        save_path=os.path.join(models_dir, "tfidf_vectorizer.pkl"),
    )

    return df, profiles, matcher, recommender, extractor


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/115px-Python-logo-notext.svg.png",
                 width=50)
        st.markdown("## 🎯 Skill Gap Analyzer")
        st.markdown("---")
        st.markdown("""
        **How it works:**
        1. Enter your current skills
        2. Select a target job role
        3. Click **Analyze**
        4. View your gap analysis
        5. Get a learning path
        """)
        st.markdown("---")
        st.markdown("**Tech Stack:**")
        st.markdown("Python · Pandas · Scikit-learn · TF-IDF · Streamlit")
        st.markdown("---")
        st.caption("BCA ML Project · Skill Gap Analyzer v1.0")


# ── Input section ─────────────────────────────────────────────────────────────

def render_input_section():
    st.markdown('<p class="main-header">🎯 Skill Gap Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand the gap between your skills and your dream job.</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.subheader("📝 Your Current Skills")
        skills_input = st.text_area(
            "Enter your skills (one per line or comma-separated):",
            placeholder="e.g. Python, SQL, Pandas, Excel\nor one skill per line",
            height=160,
            key="skills_input",
        )
        st.caption("💡 Tip: Use common skill names. Variations like 'ML' or 'sklearn' are handled automatically.")

    with col2:
        st.subheader("🎯 Target Job Role")
        target_role = st.selectbox(
            "Select a role:",
            options=SUPPORTED_ROLES,
            index=0,
            key="role_select",
        )
        st.markdown("")
        analyze_clicked = st.button("🔍 Analyze My Skills", type="primary",
                                    use_container_width=True)

    return skills_input, target_role, analyze_clicked


# ── Parse user skills ─────────────────────────────────────────────────────────

def parse_skills(raw_input: str) -> list:
    """Parse comma-separated or newline-separated skill input."""
    if not raw_input.strip():
        return []
    # Support both comma and newline separation
    parts = []
    for line in raw_input.splitlines():
        parts.extend([s.strip() for s in line.split(",") if s.strip()])
    return parts


# ── Score cards ───────────────────────────────────────────────────────────────

def render_score_cards(score: ScoreResult):
    st.markdown("### 📊 Your Skill Gap Scores")
    c1, c2, c3, c4 = st.columns(4)

    readiness_color = (
        "#22c55e" if score.readiness_score >= 70 else
        "#f59e0b" if score.readiness_score >= 40 else
        "#ef4444"
    )
    gap_color = (
        "#ef4444" if score.gap_score >= 60 else
        "#f59e0b" if score.gap_score >= 30 else
        "#22c55e"
    )

    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:{readiness_color}">{score.readiness_score}%</div>
            <div class="score-label">Readiness Score</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:{gap_color}">{score.gap_score}%</div>
            <div class="score-label">Skill Gap Score</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:#3b82d4">{score.skill_match_pct}%</div>
            <div class="score-label">Skills Matched</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:#7c5cd8">{score.readiness_label}</div>
            <div class="score-label">Your Level</div>
        </div>""", unsafe_allow_html=True)


# ── Skill breakdown chips ─────────────────────────────────────────────────────

def render_skill_chips(match_result):
    st.markdown("### 🔍 Skill Breakdown")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**✅ Matched ({len(match_result.matched)})**")
        if match_result.matched:
            chips = " ".join(
                f'<span class="skill-chip matched-chip">{s.skill}</span>'
                for s in match_result.matched
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("None matched")

    with col2:
        st.markdown(f"**🟡 Partial ({len(match_result.partial)})**")
        if match_result.partial:
            chips = " ".join(
                f'<span class="skill-chip partial-chip">{s.skill}</span>'
                for s in match_result.partial
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("None")

    with col3:
        st.markdown(f"**❌ Missing ({len(match_result.missing)})**")
        if match_result.missing:
            # Sort by importance
            sorted_missing = sorted(match_result.missing,
                                    key=lambda x: ["high", "medium", "low"].index(x.importance))
            chips = " ".join(
                f'<span class="skill-chip missing-chip">{s.skill} [{s.importance[0].upper()}]</span>'
                for s in sorted_missing
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.success("No missing skills! 🎉")


# ── Charts ────────────────────────────────────────────────────────────────────

def render_charts(score: ScoreResult, match_result):
    col1, col2 = st.columns(2)

    with col1:
        # Donut chart: matched vs partial vs missing
        labels = ["Matched", "Partial", "Missing"]
        values = [score.n_matched, score.n_partial, score.n_missing]
        colors = ["#22c55e", "#f59e0b", "#ef4444"]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker_colors=colors,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} skills<extra></extra>",
        )])
        fig.update_layout(
            title="Skill Coverage Overview",
            showlegend=True,
            height=320,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Category readiness bar chart
        if score.category_breakdown:
            cats   = list(score.category_breakdown.keys())
            pcts   = [score.category_breakdown[c]["readiness_pct"] for c in cats]
            colors_cat = [
                "#22c55e" if p >= 70 else
                "#f59e0b" if p >= 40 else
                "#ef4444"
                for p in pcts
            ]
            fig2 = go.Figure(go.Bar(
                x=pcts,
                y=cats,
                orientation="h",
                marker_color=colors_cat,
                text=[f"{p}%" for p in pcts],
                textposition="outside",
                hovertemplate="%{y}: %{x}% readiness<extra></extra>",
            ))
            fig2.update_layout(
                title="Readiness by Skill Category",
                xaxis=dict(range=[0, 110], title="Readiness %"),
                height=320,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ── Readiness gauge ───────────────────────────────────────────────────────────

def render_gauge(score: ScoreResult):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score.readiness_score,
        delta={"reference": 50, "increasing": {"color": "#22c55e"}},
        title={"text": f"Job Readiness — {score.target_role}", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": "#3b82d4"},
            "steps": [
                {"range": [0, 30],  "color": "#fee2e2"},
                {"range": [30, 50], "color": "#fef9c3"},
                {"range": [50, 70], "color": "#fef3c7"},
                {"range": [70, 85], "color": "#d1fae5"},
                {"range": [85, 100], "color": "#bbf7d0"},
            ],
            "threshold": {
                "line": {"color": "green", "width": 3},
                "thickness": 0.75,
                "value": 85,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)


# ── Learning path ─────────────────────────────────────────────────────────────

def render_learning_path(path: LearningPath):
    st.markdown("### 🗺️ Personalized Learning Path")
    st.info(path.summary_message)

    if not path.recommendations:
        st.success("🎉 You already have all the required skills for this role!")
        return

    priority_order = ["high", "deepen", "medium", "low"]
    priority_headers = {
        "high":   "🔴 HIGH PRIORITY — Critical Missing Skills",
        "deepen": "🟡 DEEPEN — Strengthen Partial Skills",
        "medium": "🟠 MEDIUM PRIORITY — Important Skills",
        "low":    "🟢 LOW PRIORITY — Nice-to-Have Skills",
    }

    for p in priority_order:
        recs = [r for r in path.recommendations if r.priority == p]
        if not recs:
            continue

        st.markdown(f"#### {priority_headers[p]}")
        for rec in recs:
            border_color = PRIORITY_COLORS.get(p, "#3b82d4")
            with st.container():
                st.markdown(f"""
                <div class="rec-card" style="border-left-color:{border_color}">
                    <b>#{rec.rank} {rec.skill.title()}</b>
                    {"<br><small>" + rec.description + "</small>" if rec.description else ""}
                    <br><small style="color:#57606a">{rec.reason}</small>
                </div>
                """, unsafe_allow_html=True)

                if rec.resources:
                    for res in rec.resources[:2]:
                        st.markdown(
                            f"&nbsp;&nbsp;&nbsp;→ **[{res.title}]({res.url})** "
                            f"*({res.type.upper()} · {res.duration})*"
                        )
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Alternative roles
    if path.alternative_roles:
        st.markdown("#### 💡 Alternative Roles You Might Be Closer To")
        alt_cols = st.columns(len(path.alternative_roles))
        for i, (role, sim) in enumerate(path.alternative_roles):
            with alt_cols[i]:
                st.metric(label=role, value=f"{sim:.0%}", delta="NLP similarity")


# ── EDA tab ───────────────────────────────────────────────────────────────────

def render_eda_tab(df: pd.DataFrame):
    st.markdown("### 📊 Dataset Explorer")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records",  len(df))
    col2.metric("Unique Roles",   df["role"].nunique())
    col3.metric("Unique Skills",  df["skill"].nunique())

    st.markdown("#### Top 20 Most In-Demand Skills")
    top_skills = df["skill"].value_counts().head(20)
    fig = px.bar(top_skills, orientation="h",
                 labels={"value": "# Roles", "index": "Skill"},
                 color=top_skills.values,
                 color_continuous_scale="Blues")
    fig.update_layout(height=450, showlegend=False,
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Skills per Role")
    role_counts = df.groupby("role")["skill"].count().sort_values(ascending=True)
    fig2 = px.bar(role_counts, orientation="h",
                  labels={"value": "# Skills", "index": "Role"},
                  color=role_counts.values,
                  color_continuous_scale="Purples")
    fig2.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Raw Dataset (filtered)")
    role_filter = st.selectbox("Filter by role:", ["All"] + SUPPORTED_ROLES, key="eda_role")
    filtered = df if role_filter == "All" else df[df["role"] == role_filter]
    st.dataframe(filtered[["role", "skill", "importance", "category"]],
                 use_container_width=True, height=300)


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Load pipeline
    df, profiles, matcher, recommender, extractor = load_pipeline()

    # Navigation tabs
    tab_analyze, tab_eda = st.tabs(["🔍 Analyze Skills", "📊 Explore Dataset"])

    with tab_analyze:
        skills_input, target_role, analyze_clicked = render_input_section()

        if analyze_clicked:
            raw_skills = parse_skills(skills_input)

            if not raw_skills:
                st.warning("⚠️ Please enter at least one skill before analyzing.")
                st.stop()

            if len(raw_skills) < 1:
                st.warning("⚠️ Enter at least one skill.")
                st.stop()

            with st.spinner("Analyzing your skill profile…"):
                # Normalize
                aliases       = load_aliases(ALIASES_JSON)
                user_skills   = normalize_user_skills(raw_skills, aliases)

                # Match
                match_result  = matcher.match(raw_skills, target_role)

                # NLP similarity
                nlp_sims      = extractor.user_role_similarity(user_skills)
                target_sim    = nlp_sims.get(target_role, None)

                # Score
                score_result  = compute_scores(match_result, nlp_similarity=target_sim)

                # Recommend
                learning_path = recommender.recommend(match_result, score_result, nlp_sims)

            # ── Display ───────────────────────────────────────────────────────
            st.success(f"✅ Analysis complete for **{target_role}**")
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # Score cards
            render_score_cards(score_result)
            st.markdown("")

            # Gauge
            render_gauge(score_result)
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # Skill chips
            render_skill_chips(match_result)
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # Charts
            render_charts(score_result, match_result)
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # NLP similarity
            if nlp_sims:
                st.markdown("### 🤖 NLP Similarity Scores (TF-IDF)")
                st.caption(
                    "These scores measure how closely your skill profile matches each role's "
                    "job description using TF-IDF cosine similarity. This is the ML/NLP component."
                )
                sim_df = pd.DataFrame(
                    sorted(nlp_sims.items(), key=lambda x: -x[1]),
                    columns=["Role", "NLP Similarity"]
                )
                sim_df["NLP Similarity"] = sim_df["NLP Similarity"].apply(lambda x: f"{x:.2%}")
                st.dataframe(sim_df, use_container_width=True, hide_index=True)
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # Learning path
            render_learning_path(learning_path)

        else:
            st.markdown("---")
            st.info("👆 Enter your skills and select a target role, then click **Analyze My Skills** to get started.")

            # Example showcase
            with st.expander("📌 See an example"):
                st.markdown("""
                **User Profile:**
                - Skills: Python, SQL, Pandas, Excel

                **Target Role:** Data Scientist

                **Expected Output:**
                - Matched: Python, SQL, Pandas
                - Missing: Machine Learning, Statistics, NumPy, Scikit-learn, Deep Learning…
                - Readiness Score: ~25%
                - Learning Path: 10+ prioritized skills with resources
                """)

    with tab_eda:
        render_eda_tab(df)


if __name__ == "__main__":
    main()
