"""
Sentiment Analysis Model Training Pipeline
============================================
Trains a text sentiment classifier using scikit-learn.

Usage:
    # Full training with Sentiment140 dataset:
    python train.py --data path/to/sentiment140.csv

    # Demo training with embedded sample data:
    python train.py --demo

The trained model and vectorizer are serialized to:
    - model/sentiment_model.pkl
    - model/tfidf_vectorizer.pkl
    - model/metadata.json
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DEMO_DATA = {
    "text": [
        # Positive samples
        "I absolutely love this product, it works perfectly!",
        "This is the best experience I've ever had, truly amazing",
        "Fantastic service, I'm so happy with the results",
        "Great quality and fast delivery, highly recommend",
        "I'm thrilled with my purchase, exceeded expectations",
        "Beautiful design and works like a charm, love it",
        "Wonderful customer support, they resolved my issue quickly",
        "So impressed with the quality, definitely worth the price",
        "Had an amazing time, the staff was incredibly friendly",
        "Excellent product, I've recommended it to all my friends",
        "This made my day so much better, thank you!",
        "Incredible value for money, best purchase this year",
        "Really happy with how things turned out, great job",
        "The team did an outstanding job, very professional",
        "Can't say enough good things about this company",
        "Top notch quality, will definitely be ordering again",
        "Such a pleasant surprise, way better than expected",
        "I'm genuinely impressed, this is a game changer",
        "Perfect in every way, couldn't ask for more",
        "Five stars all around, absolutely brilliant experience",
        "My kids loved it too, great for the whole family",
        "Super easy to use and the results are phenomenal",
        "Blown away by how good this is, just wow",
        "Finally found something that actually works, so relieved",
        "The upgrade was totally worth it, massive improvement",
        "Everything about this was delightful and smooth",
        "Hands down the best decision I made this month",
        "I keep coming back because the quality never drops",
        "So grateful for the quick turnaround, lifesaver!",
        "Honestly exceeded every expectation I had going in",
        "This deserves way more attention, criminally underrated",
        "Made with real care, you can tell the difference",
        "Feels premium without the premium price tag, love that",
        "Customer for life after this experience, no question",
        "The attention to detail here is next level",
        "Smooth, fast, reliable — everything you want",
        "I smiled the entire time using this, pure joy",
        "Worth every single penny and then some",
        "Restored my faith in good products out there",
        "A+ all around, from packaging to performance",
        # Negative samples
        "Terrible product, complete waste of money",
        "I'm extremely disappointed with the quality, very poor",
        "Worst customer service ever, they were rude and unhelpful",
        "This broke after one day, absolutely horrible quality",
        "Don't buy this, it's a complete scam and ripoff",
        "Very frustrating experience, nothing works as advertised",
        "I regret purchasing this, total waste of time",
        "The product arrived damaged and nobody helped me",
        "Awful experience from start to finish, never again",
        "Completely useless, I want a full refund immediately",
        "This is garbage, don't waste your hard earned money",
        "So disappointed, the quality has really gone downhill",
        "Waited two weeks for this junk, unacceptable service",
        "The worst purchase I've made in years, horrible",
        "Nothing works as described, feels like false advertising",
        "Cheap materials and poor construction, fell apart fast",
        "I wouldn't recommend this to my worst enemy",
        "Support team was dismissive and unhelpful, terrible",
        "A complete disaster from order to delivery",
        "Overpriced for what you get, not worth a fraction",
        "Absolutely furious, this is the last time I order here",
        "The quality control is nonexistent, embarrassing product",
        "Had to return it immediately, completely defective",
        "Three emails and still no response, awful service",
        "Packaging was destroyed and the item was cracked",
        "How is this even legal to sell, pure junk",
        "Misleading photos and description, nothing like advertised",
        "Regret not reading reviews first, total letdown",
        "Never again, fool me once shame on you",
        "They clearly don't care about their customers at all",
        "The app crashes constantly, barely functional",
        "Paid for express and it took two weeks, joke",
        "Flimsy, cheap, and ugly — not worth a dollar",
        "I've seen better quality at a dollar store",
        "Lost my trust completely after this experience",
        "Instructions were wrong and parts were missing",
        "This is what happens when companies cut corners",
        "An insult to anyone who pays full price for this",
        "Uninstalled after ten minutes, waste of storage space",
        "Zero stars if I could, absolutely dreadful",
        # Neutral samples
        "The product is okay, nothing special but it works",
        "It's decent for the price, meets basic expectations",
        "Average quality, not great but not terrible either",
        "It does what it says, nothing more nothing less",
        "Received the item on time, seems standard quality",
        "It's fine I guess, about what I expected for this",
        "Not bad not good, just a regular everyday product",
        "Works as described, pretty standard and unremarkable",
        "Middle of the road quality, you get what you pay for",
        "It's alright, serves its purpose but nothing exciting",
        "Shipped on time, packaging was standard, product is meh",
        "Does the job but nothing to write home about",
        "Adequate for the price point, no complaints really",
        "It's functional, meets the minimum requirements I had",
        "Neither impressed nor disappointed, just plain average",
        "Got what I ordered, no surprises either way honestly",
        "Fair enough for a budget option, can't complain much",
        "It exists and it works, that's about it really",
        "Looks the same as the photo, performs as listed",
        "Standard product, standard delivery, standard everything",
    ],
    "label": [1]*40 + [0]*40 + [2]*20,
}

LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}
REVERSE_LABEL_MAP = {"negative": 0, "positive": 1, "neutral": 2}


def clean_text(text: str) -> str:
    """Preprocess text for model input."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_sentiment140(filepath: str) -> pd.DataFrame:
    """
    Load Sentiment140 dataset.
    Download from: https://www.kaggle.com/datasets/kazanova/sentiment140
    Expected CSV columns: target, id, date, flag, user, text
    Target: 0 = negative, 2 = neutral, 4 = positive
    """
    cols = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(filepath, encoding="latin-1", names=cols)

    label_remap = {0: 0, 2: 2, 4: 1}
    df["label"] = df["target"].map(label_remap)
    df["text"] = df["text"].apply(clean_text)
    df = df[["text", "label"]].dropna()

    return df


def load_demo_data() -> pd.DataFrame:
    """Load embedded demo dataset."""
    df = pd.DataFrame(DEMO_DATA)
    df["text"] = df["text"].apply(clean_text)
    return df


def train_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """Train sentiment classifier and return model artifacts + metrics."""
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    start = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s")

    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)

    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys()) if i in np.unique(y)]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "training_time_seconds": round(train_time, 2),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    print(f"\n{'='*50}")
    print(f"  Model Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1_weighted']:.4f}")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"{'='*50}\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
    }


def save_artifacts(artifacts: dict, output_dir: str = ".") -> None:
    """Serialize model, vectorizer, and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "sentiment_model.pkl")
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    metadata_path = os.path.join(output_dir, "metadata.json")

    joblib.dump(artifacts["model"], model_path)
    joblib.dump(artifacts["vectorizer"], vectorizer_path)

    metadata = {
        "model_type": "LogisticRegression",
        "vectorizer_type": "TfidfVectorizer",
        "label_map": LABEL_MAP,
        "metrics": artifacts["metrics"],
        "trained_at": datetime.now().isoformat(),
        "sklearn_version": __import__("sklearn").__version__,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to:      {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Metadata saved to:   {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--data", type=str, help="Path to Sentiment140 CSV")
    parser.add_argument("--demo", action="store_true", help="Use embedded demo data")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    if args.demo:
        print("Loading demo dataset...")
        df = load_demo_data()
    elif args.data:
        print(f"Loading Sentiment140 from {args.data}...")
        df = load_sentiment140(args.data)
    else:
        print("No data source specified. Use --demo or --data <path>")
        return

    print(f"Dataset: {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    artifacts = train_model(df)
    save_artifacts(artifacts, output_dir=args.output)


if __name__ == "__main__":
    main()
