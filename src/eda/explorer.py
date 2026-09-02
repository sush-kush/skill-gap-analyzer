"""
src/eda/explorer.py

Stage 4 — Exploratory Data Analysis (EDA)
------------------------------------------
Generates statistical summaries and visualizations of the processed
role-skill dataset so we understand the data before modeling.

Charts produced (saved to reports/):
  1. Top 20 most common skills across all roles
  2. Skill count per job role
  3. Importance tier distribution (high / medium / low)
  4. Skill category breakdown heatmap (role × category)

Why EDA matters:
  - Confirms the dataset is balanced and meaningful
  - Reveals which skills are universally demanded (e.g. Python, Git)
  - Helps tune importance weights in the scoring engine
  - Documents data quality for academic submission
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/script use

# ── Resolve paths ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

PROCESSED_CSV = os.path.join(_ROOT, "data", "processed", "processed_roles.csv")
REPORTS_DIR = os.path.join(_ROOT, "reports")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> str:
    """Save a matplotlib figure to reports/ and return the path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def load_processed(csv_path: str = PROCESSED_CSV) -> pd.DataFrame:
    """Load the processed role-skill CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {csv_path}.\n"
            "Run: python src/preprocessing/data_cleaner.py"
        )
    return pd.read_csv(csv_path)


# ── Individual chart functions ───────────────────────────────────────────────

def plot_top_skills(df: pd.DataFrame, top_n: int = 20) -> str:
    """
    Bar chart: Top N most frequently required skills across all job roles.

    Insight: Skills appearing in many roles are universally valuable —
    these should be prioritized in learning recommendations.
    """
    skill_counts = df["skill"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(skill_counts.index[::-1], skill_counts.values[::-1],
                   color="#3b82d4")
    ax.set_xlabel("Number of Roles Requiring This Skill", fontsize=11)
    ax.set_title(f"Top {top_n} Most In-Demand Skills Across All Roles", fontsize=13)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "top_skills.png")


def plot_skills_per_role(df: pd.DataFrame) -> str:
    """
    Horizontal bar chart: number of required skills per job role.

    Insight: Roles with more required skills need longer preparation time.
    """
    role_counts = df.groupby("role")["skill"].count().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(role_counts.index, role_counts.values, color="#7c5cd8")
    ax.set_xlabel("Number of Required Skills", fontsize=11)
    ax.set_title("Skill Count per Job Role", fontsize=13)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "skills_per_role.png")


def plot_importance_distribution(df: pd.DataFrame) -> str:
    """
    Stacked bar chart: importance tier breakdown per role.

    Insight: Shows how many high/medium/low priority skills each role demands.
    """
    pivot = (
        df.groupby(["role", "importance"])["skill"]
        .count()
        .unstack(fill_value=0)
    )
    # Ensure all columns exist
    for col in ["high", "medium", "low"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["high", "medium", "low"]]

    colors = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = pd.Series([0] * len(pivot), index=pivot.index)
    for tier in ["high", "medium", "low"]:
        ax.bar(pivot.index, pivot[tier], bottom=bottom,
               label=tier.capitalize(), color=colors[tier])
        bottom += pivot[tier]

    ax.set_ylabel("Number of Skills", fontsize=11)
    ax.set_title("Skill Importance Tier Distribution per Job Role", fontsize=13)
    ax.legend(title="Importance")
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "importance_distribution.png")


def plot_category_heatmap(df: pd.DataFrame) -> str:
    """
    Heatmap: skill category count per role (role × category matrix).

    Insight: Quickly shows which domains (ml, programming, devops…) each role
    emphasizes — useful for understanding role profiles.
    """
    pivot = (
        df.groupby(["role", "category"])["skill"]
        .count()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Skill Category Distribution Across Job Roles (Heatmap)", fontsize=13)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color="white" if val > 3 else "black")

    fig.colorbar(im, ax=ax, label="Skill Count")
    fig.tight_layout()

    return _save(fig, "category_heatmap.png")


# ── Summary statistics ───────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a concise statistical summary of the dataset."""
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total records       : {len(df)}")
    print(f"  Unique job roles    : {df['role'].nunique()}")
    print(f"  Unique skills       : {df['skill'].nunique()}")
    print(f"  Unique categories   : {df['category'].nunique()}")
    print()

    print("  Importance distribution:")
    for tier, count in df["importance"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {tier:<8} : {count:>4} skills  ({pct:.1f}%)")
    print()

    print("  Skills per role (summary):")
    counts = df.groupby("role")["skill"].count()
    print(f"    Min  : {counts.min()}")
    print(f"    Max  : {counts.max()}")
    print(f"    Mean : {counts.mean():.1f}")
    print()

    print("  Top 10 most in-demand skills:")
    top = df["skill"].value_counts().head(10)
    for skill, count in top.items():
        print(f"    {skill:<35} : {count} roles")
    print("=" * 55)


# ── Run all EDA ───────────────────────────────────────────────────────────────

def run_eda(csv_path: str = PROCESSED_CSV) -> dict:
    """
    Run the full EDA pipeline.

    Returns
    -------
    dict
        Paths to all generated chart files
    """
    df = load_processed(csv_path)
    print_summary(df)

    print("\nGenerating charts...")
    paths = {
        "top_skills": plot_top_skills(df),
        "skills_per_role": plot_skills_per_role(df),
        "importance_distribution": plot_importance_distribution(df),
        "category_heatmap": plot_category_heatmap(df),
    }
    print("\nAll EDA charts saved to reports/")
    return paths


if __name__ == "__main__":
    run_eda()
