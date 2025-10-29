"""Main script: download dataset (if missing), train a simple SMS spam classifier,
evaluate on a held-out test set, and save the pipeline to disk.

Usage:
  python main.py --data data/sms_spam_no_header.csv --out models/final_model.pkl

The script uses scikit-learn (TF-IDF + MultinomialNB) for a fast baseline.
"""
from pathlib import Path
import argparse
import logging
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATA_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def download_dataset(url: str, dest: Path) -> None:
    """Download dataset from URL to dest. If file exists, do nothing."""
    if dest.exists():
        logging.info(f"Dataset already exists at {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading dataset from {url} to {dest} ...")
    try:
        import requests

        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        logging.info("Download complete.")
    except Exception as exc:
        logging.error("Failed to download dataset: %s", exc)
        raise


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # remove urls
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    # replace numbers with a token
    s = re.sub(r"\d+", " <NUM> ", s)
    # remove punctuation (keep basic word separators)
    s = re.sub(r"[^\w\s<>]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_data(path: Path) -> pd.DataFrame:
    # CSV has no header: assume two columns (label, message)
    df = pd.read_csv(path, header=None, names=["label", "message"], encoding="utf-8")
    # Normalize label text
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    # Keep only expected labels
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["message"] = df["message"].fillna("").astype(str).map(clean_text)
    return df


def build_pipeline() -> Pipeline:
    vect = TfidfVectorizer(max_df=0.9, ngram_range=(1, 2), stop_words="english")
    clf = MultinomialNB()
    pipeline = Pipeline([("tfidf", vect), ("clf", clf)])
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sms_spam_no_header.csv")
    parser.add_argument("--out", default="models/final_model.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_path = Path(args.data)
    model_path = Path(args.out)

    # Download dataset if missing
    if not data_path.exists():
        download_dataset(DATA_URL, data_path)

    logging.info("Loading data from %s", data_path)
    df = load_data(data_path)
    logging.info("Loaded %d rows (spam=%d, ham=%d)", len(df), (df.label == "spam").sum(), (df.label == "ham").sum())

    X = df["message"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    pipeline = build_pipeline()
    logging.info("Training baseline pipeline (TF-IDF + MultinomialNB)...")
    pipeline.fit(X_train, y_train)

    logging.info("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    logging.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logging.info("Saved pipeline to %s", model_path)


if __name__ == "__main__":
    main()
