import os
import re
from typing import Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Data loading and cleaning
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(csv_path)

    # Normalize column names and find required fields
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get("title")
    desc_col = cols.get("course_de") or cols.get("course_description")
    if not title_col or not desc_col:
        raise ValueError("CSV must have 'Title' and either 'course_de' or 'course_description'.")

    # Coerce to strings and strip whitespace
    df[title_col] = df[title_col].astype(str).str.strip()
    df[desc_col] = df[desc_col].astype(str).str.strip()

    # Treat literal placeholders as empty
    bad = {"none", "nan", "null"}
    df[title_col] = df[title_col].apply(lambda s: "" if s.lower() in bad else s)
    df[desc_col] = df[desc_col].apply(lambda s: "" if s.lower() in bad else s)

    # Strictly drop rows where either title OR description is empty
    df = df[(df[title_col] != "") & (df[desc_col] != "")].copy()

    # Build combined text and ensure non-empty
    df["text_raw"] = (df[title_col] + " " + df[desc_col]).str.strip()
    df = df[df["text_raw"].str.len() > 0].copy()

    # Coerce numeric columns (if present)
    for col in ["Ratings", "Review Co"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, title_col, desc_col


# Text preprocessing and model
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_resource(show_spinner=False)
def build_model(df: pd.DataFrame):
    texts = df["text_raw"].apply(clean_text).tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def rank(query: str, vectorizer, X, df: pd.DataFrame, k: int):
    q = clean_text(query)
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X).flatten()
    out = df.copy()
    out["similarity"] = sims
    return out.sort_values("similarity", ascending=False).head(k)



# Streamlit UI
def ui():
    st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")
    st.title("Interactive Course Recommender 🎓")
    st.caption("Similarity over course titles and descriptions.")

    # Sidebar controls
    csv_path = st.sidebar.text_input("CSV path", value="coursera_courses.csv")
    top_k = st.sidebar.slider("Top K", min_value=5, max_value=30, value=10, step=1)
    min_rating = st.sidebar.text_input("Min rating (optional)", value="")
    difficulty = st.sidebar.selectbox("Difficulty filter", options=["", "Beginner", "Intermediate", "Advanced"])
    ctype = st.sidebar.selectbox("Type filter", options=["", "Professional Certificate", "Specialization", "Course"])
    st.sidebar.markdown("---")
    st.sidebar.write("Upload a different CSV")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    # Load data
    try:
        if uploaded is not None:
            df, title_col, desc_col = load_data(uploaded)
        else:
            df, title_col, desc_col = load_data(csv_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # Optional filters applied before model
    df_view = df.copy()

    # Strictly require title/desc non-empty (safety if a different CSV is uploaded)
    df_view = df_view[(df_view[title_col].str.len() > 0) & (df_view[desc_col].str.len() > 0)].copy()

    # Min rating with NaN guard
    if min_rating and "Ratings" in df_view.columns:
        try:
            thr = float(min_rating)
            df_view["Ratings_num"] = pd.to_numeric(df_view["Ratings"], errors="coerce")
            df_view = df_view[df_view["Ratings_num"].notna() & (df_view["Ratings_num"] >= thr)]
        except ValueError:
            st.warning("Min rating must be numeric, skipping rating filter.")

    # Difficulty and type filters
    if difficulty and "Difficulty" in df_view.columns:
        df_view = df_view[df_view["Difficulty"].astype(str).str.contains(difficulty, case=False, na=False)]
    if ctype and "Type" in df_view.columns:
        df_view = df_view[df_view["Type"].astype(str).str.contains(ctype, case=False, na=False)]

    if df_view.empty:
        st.warning("No rows after filters. Adjust filters.")
        return

    # Build model on filtered data
    vectorizer, X = build_model(df_view)

    # Query box
    default_query = "Beginner Python data analytics"
    query = st.text_input("Describe the course being searched", value=default_query)
    go = st.button("Find Courses")

    # Results
    if go and query.strip():
        ranked = rank(query, vectorizer, X, df_view, top_k)

        # Columns to display
        show_cols = [title_col, desc_col, "similarity"]
        for c in ["Ratings", "Review Co", "Difficulty", "Type", "Duration", "course_url", "Organizati"]:
            if c in ranked.columns and c not in show_cols:
                show_cols.append(c)

        # Remove rows that have empty/placeholder values in any of the shown columns
        placeholders = {"", "none", "nan", "null"}

        def is_bad(v: object) -> bool:
            if pd.isna(v):
                return True
            s = str(v).strip().lower()
            return s in placeholders

        mask_good = ~ranked[show_cols].applymap(is_bad).any(axis=1)
        ranked = ranked[mask_good].copy()

        # Shorten description and format similarity
        def shorten(s: str, n: int = 220) -> str:
            s = str(s)
            return s if len(s) <= n else s[:n] + "..."

        ranked_display = ranked.copy()
        ranked_display[desc_col] = ranked_display[desc_col].apply(lambda s: shorten(s, 220))
        ranked_display["similarity"] = ranked_display["similarity"].map(lambda x: f"{float(x):.4f}")

        st.subheader("Top Recommendations")
        st.dataframe(ranked_display[show_cols], use_container_width=True)

        # Details (only for kept rows)
        st.subheader("Details")
        for _, r in ranked.iterrows():
            with st.expander(f"{r[title_col]}  —  score {r['similarity']:.4f}"):
                st.write(r[desc_col])
                meta_lines = []
                for c in ["Ratings", "Review Co", "Difficulty", "Type", "Duration", "Organizati"]:
                    if c in ranked.columns and not is_bad(r.get(c, "")):
                        meta_lines.append(f"{c}: {r[c]}")
                if meta_lines:
                    st.markdown("\n\n".join(meta_lines))
                if "course_url" in ranked.columns and not is_bad(r.get("course_url", "")):
                    st.markdown(f"[Open course link]({r['course_url']})")

    with st.sidebar.expander("About"):
        st.write("This app uses similarity to match a free‑text query against course titles and descriptions.")
        st.write("Great for demonstrating classic NLP retrieval without external APIs.")


if __name__ == "__main__":
    ui()
