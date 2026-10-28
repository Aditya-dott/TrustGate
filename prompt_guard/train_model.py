"""
Train the Prompt Guard classifier.

Steps:
    1. Load the generated dataset.
    2. Build an embedding backend (SentenceTransformer preferred; falls back
       to a local HashingVectorizer + TruncatedSVD pipeline if HF is
       unreachable).
    3. Fit the embedding backend on the training corpus (no-op for
       SentenceTransformer, non-trivial for LocalHashSvd).
    4. Encode train and test sets.
    5. Fit a calibrated multinomial logistic regression head on top.
    6. Evaluate per-class and binary (malicious-vs-benign) metrics.
    7. Persist embedding backend, classifier, and metadata to `models/`.

Usage:
    python -m prompt_guard.train_model                # prefers SentenceTransformer
    python -m prompt_guard.train_model --backend local  # forces the offline path

The classifier is 5-way over:
    benign, prompt_injection, jailbreak, data_exfiltration, model_extraction

Any label != 'benign' is treated as malicious at inference time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Allow running as `python prompt_guard/train_model.py` from project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_guard.embedding_backend import (  # noqa: E402
    EmbeddingBackend, LocalHashSvdBackend, SentenceTransformerBackend, make_backend,
)

DATA_PATH = ROOT / "dataset" / "prompts_dataset.csv"
MODELS_DIR = ROOT / "models"

LABELS = ["benign", "prompt_injection", "jailbreak", "data_exfiltration", "model_extraction"]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["prompt", "label"]).reset_index(drop=True)
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"].str.len() > 3].reset_index(drop=True)
    return df


def build_backend(preference: str) -> EmbeddingBackend:
    if preference == "auto":
        return make_backend(prefer="sentence_transformer")
    if preference == "sentence_transformer":
        return SentenceTransformerBackend()
    if preference == "local":
        return LocalHashSvdBackend()
    raise ValueError(f"Unknown backend preference: {preference}")


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    base = LogisticRegression(
        C=4.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf: CalibratedClassifierCV, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    report_txt = classification_report(y_test, y_pred, labels=LABELS, digits=3, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    y_bin_true = (y_test != "benign").astype(int)
    y_bin_pred = (y_pred != "benign").astype(int)
    tp = int(((y_bin_true == 1) & (y_bin_pred == 1)).sum())
    tn = int(((y_bin_true == 0) & (y_bin_pred == 0)).sum())
    fp = int(((y_bin_true == 0) & (y_bin_pred == 1)).sum())
    fn = int(((y_bin_true == 1) & (y_bin_pred == 0)).sum())
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "report": report_txt,
        "confusion_matrix": cm.tolist(),
        "binary": {
            "accuracy": acc,
            "precision_malicious": precision,
            "recall_malicious": recall,
            "f1_malicious": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend",
        choices=["auto", "sentence_transformer", "local"],
        default="auto",
        help="Embedding backend. 'auto' tries SentenceTransformer, falls back to local.",
    )
    args = ap.parse_args()

    print("Loading dataset...")
    df = load_dataset()
    print(f"  {len(df)} rows, label distribution:")
    for k, v in df["label"].value_counts().items():
        print(f"    {k:24s} {v}")

    X_text_train, X_text_test, y_train, y_test = train_test_split(
        df["prompt"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=1729,
        stratify=df["label"].tolist(),
    )
    print(f"\nTrain size: {len(X_text_train)}, Test size: {len(X_text_test)}")

    print(f"\nBuilding embedding backend (preference={args.backend})...")
    t0 = time.time()
    backend = build_backend(args.backend)
    print(f"  chose: {backend.name}")

    print("Fitting embedding backend on training corpus (no-op for pretrained)...")
    backend.fit(X_text_train)
    print(f"  ready in {time.time() - t0:.1f}s")

    print("\nEncoding train set...")
    t0 = time.time()
    X_train = backend.encode(X_text_train)
    print(f"  encoded {X_train.shape} in {time.time() - t0:.1f}s")

    print("Encoding test set...")
    t0 = time.time()
    X_test = backend.encode(X_text_test)
    print(f"  encoded {X_test.shape} in {time.time() - t0:.1f}s")

    print("\nTraining classifier...")
    t0 = time.time()
    clf = train_classifier(X_train, np.asarray(y_train))
    print(f"  trained in {time.time() - t0:.1f}s")

    print("\nEvaluating...")
    results = evaluate(clf, X_test, np.asarray(y_test))
    print("\n=== Per-class report ===")
    print(results["report"])
    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print("Labels:", LABELS)
    for row in results["confusion_matrix"]:
        print(" ", row)
    print("\n=== Binary malicious-vs-benign ===")
    for k, v in results["binary"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    backend_path = MODELS_DIR / "embedding_model.pkl"
    clf_path = MODELS_DIR / "classifier.pkl"
    meta_path = MODELS_DIR / "classifier_meta.json"

    backend.save(backend_path)
    joblib.dump(clf, clf_path)
    meta = {
        "embedding_backend": backend.name,
        "labels": LABELS,
        "train_size": len(X_text_train),
        "test_size": len(X_text_test),
        "binary_metrics": results["binary"],
        "confusion_matrix": results["confusion_matrix"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved embedding backend to {backend_path}")
    print(f"Saved classifier to {clf_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
