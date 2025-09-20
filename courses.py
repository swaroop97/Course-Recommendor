import argparse
import os
import re
import sys
import json
import joblib
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL_DIR = "artifacts"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "tfidf_vectorizer.joblib")
DEFAULT_INDEX_PATH = os.path.join(DEFAULT_MODEL_DIR, "tfidf_matrix.joblib")
DEFAULT_META_PATH = os.path.join(DEFAULT_MODEL_DIR, "metadata.json")


def normalize_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get("title")
    desc_col = cols.get("course_de") or cols.get("course_description")
    if not title_col or not desc_col:
        print("Required columns not found. Expecting 'Title' and either 'course_de' or 'course_description'.")
        print("Found columns:", list(df.columns))
        sys.exit(1)
    return title_col, desc_col


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)      # remove punctuation, numbers
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_corpus(df: pd.DataFrame, title_col: str, desc_col: str) -> pd.DataFrame:
    df = df.copy()
    df[title_col] = df[title_col].astype(str).fillna("")
    df[desc_col] = df[desc_col].astype(str).fillna("")
    df["text_raw"] = (df[title_col].fillna("") + " " + df[desc_col].fillna("")).str.strip()
    df = df[df["text_raw"].str.len() > 0].copy()
    if df.empty:
        print("No valid text rows after cleaning. Check your CSV content.")
        sys.exit(1)
    df["text"] = df["text_raw"].apply(clean_text)
    return df


def train_vectorizer(corpus: List[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2, ngram_range=(1, 2))
    vectorizer.fit(corpus)
    return vectorizer


def vectorize(vectorizer: TfidfVectorizer, corpus: List[str]):
    return vectorizer.transform(corpus)


def save_artifacts(vectorizer: TfidfVectorizer, tfidf_matrix, df_meta: pd.DataFrame, args):
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, DEFAULT_MODEL_PATH)
    joblib.dump(tfidf_matrix, DEFAULT_INDEX_PATH)
    meta = {
        "row_count": int(df_meta.shape[0]),
        "columns": df_meta.columns.tolist(),
        "title_col": args.title_col,
        "desc_col": args.desc_col,
        "csv_path": args.csv_path
    }
    with open(DEFAULT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_artifacts():
    if not (os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(DEFAULT_INDEX_PATH) and os.path.exists(DEFAULT_META_PATH)):
        print("Artifacts not found. Please run with --fit to train and save the model first.")
        sys.exit(1)
    vectorizer = joblib.load(DEFAULT_MODEL_PATH)
    tfidf_matrix = joblib.load(DEFAULT_INDEX_PATH)
    with open(DEFAULT_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return vectorizer, tfidf_matrix, meta


def rank_courses(query: str, vectorizer: TfidfVectorizer, tfidf_matrix, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    q = clean_text(query)
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    out = df.copy()
    out["similarity"] = sims
    out = out.sort_values("similarity", ascending=False).head(top_k)
    return out


def pretty_print(rows: pd.DataFrame, title_col: str, desc_col: str):
    opt_cols = []
    for c in ["Ratings", "Difficulty", "Type", "Duration", "Review Co", "course_url"]:
        if c in rows.columns:
            opt_cols.append(c)

    for _, r in rows.iterrows():
        title = str(r[title_col])
        desc = str(r[desc_col])[:800]
        sim = float(r["similarity"])
        print(f"Course Title: {title}")
        print(f"Description: {desc}")
        for c in opt_cols:
            print(f"{c}: {r[c]}")
        print(f"Similarity Score: {sim:.4f}")
        print("-" * 80)


def small_eval(queries: Dict[str, List[str]], vectorizer, tfidf_matrix, df: pd.DataFrame, title_col: str, desc_col: str, k: int = 5):
    print("\nQuick evaluation:")
    hits = 0
    total = 0
    for intent, keywords in queries.items():
        ranked = rank_courses(intent, vectorizer, tfidf_matrix, df, k)
        text_blob = " ".join((ranked[title_col] + " " + ranked[desc_col]).tolist()).lower()
        found = any(kw.lower() in text_blob for kw in keywords)
        hits += int(found)
        total += 1
        print(f"- Intent: '{intent}' | Keywords: {keywords} | Hit@{k}: {found}")
    print(f"Hit rate: {hits}/{total} = {hits/total if total else 0:.2f}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Course Recommender (TF-IDF, portfolio-ready)")
    p.add_argument("--csv_path", type=str, default="coursera_courses.csv", help="Path to CSV dataset")
    p.add_argument("--query", type=str, default="I want to learn data analytics with Python, beginner level.", help="User query")
    p.add_argument("--top_k", type=int, default=10, help="Top K results")
    p.add_argument("--fit", action="store_true", help="Train and save artifacts")
    p.add_argument("--use_saved", action="store_true", help="Load saved artifacts instead of refitting")
    p.add_argument("--title_col", type=str, default=None, help="Override title column name")
    p.add_argument("--desc_col", type=str, default=None, help="Override description column name")
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    try:
        df_raw = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"File not found: {args.csv_path}")
        sys.exit(1)

    # Resolve columns
    if args.title_col and args.desc_col:
        title_col, desc_col = args.title_col, args.desc_col
    else:
        title_col, desc_col = normalize_columns(df_raw)

    df = build_corpus(df_raw, title_col, desc_col)

    # Fit or load
    if args.use_saved:
        vectorizer, tfidf_matrix, meta = load_artifacts()
        # Reindex df to match saved order (best is to refit on same CSV to avoid mismatch)
        if os.path.abspath(meta["csv_path"]) != os.path.abspath(args.csv_path):
            print("Warning: Loaded artifacts were trained on a different CSV. Consider refitting with --fit.")
    else:
        vectorizer = train_vectorizer(df["text"].tolist())
        tfidf_matrix = vectorize(vectorizer, df["text"].tolist())
        if args.fit:
            # store column names for printing later
            args.title_col, args.desc_col = title_col, desc_col
            save_artifacts(vectorizer, tfidf_matrix, df, args)

    # Rank and print
    ranked = rank_courses(args.query, vectorizer, tfidf_matrix, df, args.top_k)
    pretty_print(ranked, title_col, desc_col)

    # Quick sanity evaluation (optional; simple heuristic)
    eval_queries = {
        "Beginner Python data analytics": ["python", "data", "analytics"],
        "Machine learning specialization": ["machine", "learning"],
        "Deep learning with TensorFlow": ["deep", "tensorflow"],
        "SQL for data engineering": ["sql", "data", "engineer"]
    }
    small_eval(eval_queries, vectorizer, tfidf_matrix, df, title_col, desc_col, k=5)


if __name__ == "__main__":
    main()

