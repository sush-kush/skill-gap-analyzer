"""
setup.py - One-time project setup script

Run this once after cloning:
    python setup.py

What it does:
  1. Creates required directories
  2. Runs data cleaning to produce processed_roles.csv
  3. Builds feature artifacts (binary + weighted matrices, vocabulary)
  4. Trains and saves the TF-IDF NLP model
  5. Runs EDA to generate charts in reports/
  6. Confirms everything is ready
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))


def ensure_dirs():
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "logs",
    ]
    for d in dirs:
        path = os.path.join(_ROOT, d)
        os.makedirs(path, exist_ok=True)
    print("[OK] Directories ready.")


def run_cleaning():
    from preprocessing.data_cleaner import clean_dataset
    print("[..] Running data cleaning...")
    df = clean_dataset()
    print("[OK] Cleaned dataset: {} records, {} roles, {} skills.".format(
        len(df), df['role'].nunique(), df['skill'].nunique()))
    return df


def run_feature_engineering(df):
    from features.feature_engineering import (
        build_role_profiles, build_skill_vocabulary,
        build_binary_matrix, build_weighted_matrix, save_features
    )
    print("[..] Building feature artifacts...")
    profiles    = build_role_profiles(df)
    vocab       = build_skill_vocabulary(df)
    bin_matrix  = build_binary_matrix(df, vocab)
    wgt_matrix  = build_weighted_matrix(df, vocab)
    save_features(bin_matrix, wgt_matrix, vocab, profiles)
    print("[OK] Features saved. Vocabulary: {} skills.".format(len(vocab)))


def run_nlp_model():
    from nlp.skill_extractor import build_and_save_extractor
    print("[..] Training TF-IDF NLP model...")
    extractor = build_and_save_extractor()
    print("[OK] NLP model saved. Vocabulary: {} terms.".format(len(extractor.vocabulary_)))


def run_eda():
    from eda.explorer import run_eda
    print("[..] Generating EDA charts...")
    run_eda()
    print("[OK] EDA charts saved to reports/.")


def main():
    print("\n" + "="*55)
    print("  Skill Gap Analyzer - Project Setup")
    print("="*55 + "\n")

    ensure_dirs()
    df = run_cleaning()
    run_feature_engineering(df)
    run_nlp_model()
    run_eda()

    print("\n" + "="*55)
    print("  [DONE] Setup complete!")
    print("="*55)
    print("\n  To run the app:")
    print("    streamlit run app/streamlit_app.py\n")
    print("  To run tests:")
    print("    python -m pytest tests/ -v\n")


if __name__ == "__main__":
    main()
