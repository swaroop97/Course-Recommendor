## Course Recommender (TF‑IDF + Streamlit)

A lightweight **NLP-based course recommender** that matches a free-text query against course titles + descriptions using **TF‑IDF** and **cosine similarity**.

- **Interactive app**: Streamlit UI (`streamcourse_app.py`; `streamlit_app.py` is a thin entry point for Streamlit Community Cloud’s default main-file name)
- **CLI script**: train/save artifacts + query from terminal (`courses.py`)

## Features

- **TF‑IDF (1–2 grams)** over title + description
- **Cosine similarity ranking** with top‑K results
- **Streamlit filters** (rating threshold, difficulty, type) when present in the CSV
- **Bring your own CSV** via file upload or path input
- **Optional persisted artifacts** (`joblib`) for reuse from the CLI

## Dataset expectations

The default dataset file is `coursera_courses.csv`.

Required columns (case-insensitive):
- `Title`
- `course_de` **or** `course_description`

Optional columns (used if present):
- `Ratings`, `Review Co`, `Difficulty`, `Type`, `Duration`, `course_url`, `Organizati`

## Quickstart (local)

### 1) Install

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Run the Streamlit app

```bash
streamlit run streamcourse_app.py
```

(`streamlit run streamlit_app.py` works the same.)

In the sidebar, the default CSV path points at the bundled `coursera_courses.csv` next to the scripts (or upload your own CSV).

### Streamlit Community Cloud

In the app settings, set **Main file** to either `streamlit_app.py` (recommended) or `streamcourse_app.py`. The app resolves the dataset path relative to the script directory so the CSV loads even when the process working directory is not the repo root.

### 3) Run via CLI (optional)

```bash
# Query directly (refits each run unless you use --use_saved)
python courses.py --csv_path coursera_courses.csv --query "Beginner Python data analytics" --top_k 10

# Fit + save artifacts
python courses.py --csv_path coursera_courses.csv --fit

# Load saved artifacts
python courses.py --csv_path coursera_courses.csv --use_saved --query "SQL for data engineering" --top_k 10
```

## Project structure

```text
.
├─ streamlit_app.py        # Streamlit Cloud entry (imports streamcourse_app)
├─ streamcourse_app.py     # Streamlit UI
├─ courses.py              # CLI trainer / ranker with artifact support
├─ coursera_courses.csv    # Dataset (default)
├─ requirements.txt
└─ README.md
```

## Notes

- The Streamlit app caches the loaded data and model for fast iteration.
- Similarity is computed over **title + description** after basic text cleaning (lowercasing, URL removal, alpha-only tokens).
