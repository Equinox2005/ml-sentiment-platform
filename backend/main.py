"""
Sentiment Analysis API
=======================
FastAPI backend serving the trained sentiment model from model/.

Run from the backend/ directory:
    uvicorn main:app --reload
"""

import json
import logging
import os
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")

# ── Config ───────────────────────────────────────────────────────────
MODEL_DIR = os.getenv(
    "MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
)
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

MAX_TEXT_LENGTH = 5000
MAX_BATCH_SIZE = 50
KEY_WORD_LIMIT = 8

# Class ids written by model/train.py
LABEL_NAMES = {0: "negative", 1: "positive", 2: "neutral"}

EMPTY_TEXT_DETAIL = (
    "Nothing left to analyze after removing links, mentions, hashtags, numbers and symbols."
)


# ── Schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, examples=["I love this product!"])


class WordScore(BaseModel):
    word: str
    score: float
    type: str


class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    inference_time_ms: float
    timestamp: str
    key_words: list[WordScore] = []


class BatchPredictRequest(BaseModel):
    texts: list[Annotated[str, Field(max_length=MAX_TEXT_LENGTH)]] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE
    )


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    skipped: list[int] = Field(
        default=[], description="Indexes of texts with nothing left to analyze after cleaning"
    )
    total_inference_time_ms: float


class CompareRequest(BaseModel):
    text_a: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    text_b: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)


class CompareResponse(BaseModel):
    result_a: PredictResponse
    result_b: PredictResponse
    delta: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_predictions: int
    uptime_seconds: int


class StatsResponse(BaseModel):
    total_predictions: int
    label_distribution: dict[str, int]
    avg_confidence: float


class ModelInfoResponse(BaseModel):
    model_type: str
    labels: list[str]
    vocabulary_size: int
    accuracy: float | None = None
    f1_weighted: float | None = None
    train_samples: int | None = None
    test_samples: int | None = None
    trained_at: str | None = None


# ── Text Preprocessing ───────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Mirror the preprocessing used during training."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Model ────────────────────────────────────────────────────────────
def polarity_weights(classifier: Any, labels: list[str]) -> np.ndarray | None:
    """Per-feature weight toward positive (> 0) or negative (< 0), if the model is linear."""
    coef = getattr(classifier, "coef_", None)
    if coef is None or not {"positive", "negative"} <= set(labels):
        return None
    if coef.shape[0] == 1:
        # Binary models store a single row: the log-odds of labels[1] over labels[0].
        return coef[0] if labels[1] == "positive" else -coef[0]
    return coef[labels.index("positive")] - coef[labels.index("negative")]


@dataclass
class SentimentModel:
    classifier: Any
    vectorizer: Any
    labels: list[str]  # label of each predict_proba column
    feature_names: np.ndarray
    polarity: np.ndarray | None
    metadata: dict

    @classmethod
    def load(cls, model_dir: str) -> "SentimentModel":
        paths = {
            name: os.path.join(model_dir, name)
            for name in ("sentiment_model.pkl", "tfidf_vectorizer.pkl", "metadata.json")
        }
        for name in ("sentiment_model.pkl", "tfidf_vectorizer.pkl"):
            if not os.path.exists(paths[name]):
                raise FileNotFoundError(
                    f"{paths[name]} not found. Train a model with `python model/train.py` first."
                )

        classifier = joblib.load(paths["sentiment_model.pkl"])
        vectorizer = joblib.load(paths["tfidf_vectorizer.pkl"])
        labels = [LABEL_NAMES[int(c)] for c in classifier.classes_]

        metadata = {}
        if os.path.exists(paths["metadata.json"]):
            with open(paths["metadata.json"]) as f:
                metadata = json.load(f)

        return cls(
            classifier=classifier,
            vectorizer=vectorizer,
            labels=labels,
            feature_names=vectorizer.get_feature_names_out(),
            polarity=polarity_weights(classifier, labels),
            metadata=metadata,
        )

    def key_words(self, features) -> list[WordScore]:
        """Terms that pushed this prediction the most (tf-idf value x polarity weight)."""
        if self.polarity is None:
            return []
        row = features.tocsr()
        scores = row.data * self.polarity[row.indices]
        top = np.argsort(-np.abs(scores))[:KEY_WORD_LIMIT]
        return [
            WordScore(
                word=str(self.feature_names[row.indices[i]]),
                score=round(abs(float(scores[i])), 4),
                type="positive" if scores[i] > 0 else "negative",
            )
            for i in top
            if scores[i] != 0
        ]


class Stats:
    """Running totals since startup. Only labels and confidences are kept, never the text."""

    def __init__(self) -> None:
        self.total = 0
        self.confidence_sum = 0.0
        self.labels: Counter[str] = Counter()

    def record(self, result: PredictResponse) -> None:
        self.total += 1
        self.confidence_sum += result.confidence
        self.labels[result.prediction] += 1


MODEL: SentimentModel | None = None
STATS = Stats()
STARTED_AT = time.monotonic()


# ── Prediction Logic ─────────────────────────────────────────────────
def require_model() -> SentimentModel:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MODEL


def analyze(model: SentimentModel, text: str) -> PredictResponse | None:
    """Run inference on one text. Returns None if nothing is left after cleaning."""
    cleaned = clean_text(text)
    if not cleaned:
        return None

    start = time.perf_counter()
    features = model.vectorizer.transform([cleaned])
    proba = model.classifier.predict_proba(features)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    best = int(np.argmax(proba))
    return PredictResponse(
        text=text,
        cleaned_text=cleaned,
        prediction=model.labels[best],
        confidence=round(float(proba[best]), 4),
        probabilities={label: round(float(p), 4) for label, p in zip(model.labels, proba)},
        inference_time_ms=round(elapsed_ms, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        key_words=model.key_words(features),
    )


# ── App ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    global MODEL
    MODEL = SentimentModel.load(MODEL_DIR)
    logger.info(
        "Loaded %s from %s (labels: %s, %d features)",
        type(MODEL.classifier).__name__,
        os.path.abspath(MODEL_DIR),
        ", ".join(MODEL.labels),
        len(MODEL.feature_names),
    )
    yield
    MODEL = None


app = FastAPI(
    title="Sentiment Analysis API",
    description="Text sentiment prediction with a TF-IDF + logistic regression model.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        total_predictions=STATS.total,
        uptime_seconds=round(time.monotonic() - STARTED_AT),
    )


@app.get("/model", response_model=ModelInfoResponse)
async def model_info():
    """Describe the loaded model and how it scored on its test split."""
    model = require_model()
    metrics = model.metadata.get("metrics", {})
    return ModelInfoResponse(
        model_type=type(model.classifier).__name__,
        labels=model.labels,
        vocabulary_size=len(model.feature_names),
        accuracy=metrics.get("accuracy"),
        f1_weighted=metrics.get("f1_weighted"),
        train_samples=metrics.get("train_samples"),
        test_samples=metrics.get("test_samples"),
        trained_at=model.metadata.get("trained_at"),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict sentiment for a single text input."""
    result = analyze(require_model(), request.text)
    if result is None:
        raise HTTPException(status_code=400, detail=EMPTY_TEXT_DETAIL)
    STATS.record(result)
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predict sentiment for up to 50 texts. Texts with nothing to analyze are skipped."""
    model = require_model()
    start = time.perf_counter()
    results, skipped = [], []
    for i, text in enumerate(request.texts):
        result = analyze(model, text)
        if result is None:
            skipped.append(i)
        else:
            STATS.record(result)
            results.append(result)

    return BatchPredictResponse(
        results=results,
        skipped=skipped,
        total_inference_time_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """Compare sentiment between two texts side by side."""
    model = require_model()
    result_a = analyze(model, request.text_a)
    result_b = analyze(model, request.text_b)
    for field, result in (("text_a", result_a), ("text_b", result_b)):
        if result is None:
            raise HTTPException(status_code=400, detail=f"{field}: {EMPTY_TEXT_DETAIL}")

    STATS.record(result_a)
    STATS.record(result_b)
    delta = {
        label: round(result_a.probabilities[label] - result_b.probabilities[label], 4)
        for label in model.labels
    }
    return CompareResponse(result_a=result_a, result_b=result_b, delta=delta)


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Prediction counts since the API started."""
    return StatsResponse(
        total_predictions=STATS.total,
        label_distribution=dict(STATS.labels),
        avg_confidence=round(STATS.confidence_sum / STATS.total, 4) if STATS.total else 0.0,
    )
