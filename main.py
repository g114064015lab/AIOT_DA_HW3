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
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.train import train_and_evaluate, save_pipeline


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


def build_and_train(data_path: Path, out_path: Path, test_size: float, random_state: int):
    logging.info("Loading data from %s", data_path)
    df = load_data(data_path)
    logging.info("Loaded %d rows (spam=%d, ham=%d)", len(df), (df.label == "spam").sum(), (df.label == "ham").sum())

    X = df["message"].values
    y = df["label"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train using src.train utilities (quick improvements)
    logging.info("Training Logistic Regression with TF-IDF features...")
    pipeline, metrics = train_and_evaluate(X_train, X_test, y_train, y_test, model_type='lr', do_grid_search=False)

    # Save pipeline
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pipeline(pipeline, out_path)
    logging.info("Saved pipeline to %s", out_path)

    # Save numeric metrics to JSON
    try:
        # metrics contains classification_report string and confusion matrix
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = pipeline.predict(X_test)
        numeric = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, pos_label='spam', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, pos_label='spam', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, pos_label='spam', zero_division=0)),
        }
        with open(out_path.parent / 'metrics.json', 'w') as fh:
            json.dump(numeric, fh)
        logging.info("Saved metrics to %s", out_path.parent / 'metrics.json')
    except Exception as e:
        logging.warning("Failed to compute/save numeric metrics: %s", e)

    # Print classification report
    print(metrics.get('classification_report', metrics.get('classification_report', '')))
    return pipeline, metrics


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

    build_and_train(data_path, model_path, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
