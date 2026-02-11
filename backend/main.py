"""
Sentiment Analysis API
=======================
FastAPI backend serving the trained sentiment model.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Globals ──────────────────────────────────────────────────────────
MODEL = None
VECTORIZER = None
LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}
PREDICTION_LOG: list[dict] = []

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))


# ── Lifespan (model loading) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    global MODEL, VECTORIZER

    model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `python model/train.py --demo` first."
        )

    MODEL = joblib.load(model_path)
    VECTORIZER = joblib.load(vectorizer_path)

    print(f"Model loaded from {model_path}")
    print(f"Vectorizer loaded from {vectorizer_path}")

    yield

    MODEL = None
    VECTORIZER = None


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time text sentiment prediction powered by scikit-learn",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["I love this product!"])


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
    texts: list[str] = Field(..., min_length=1, max_length=50)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_inference_time_ms: float


class CompareRequest(BaseModel):
    text_a: str = Field(..., min_length=1, max_length=5000)
    text_b: str = Field(..., min_length=1, max_length=5000)


class CompareResponse(BaseModel):
    result_a: PredictResponse
    result_b: PredictResponse
    delta: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_predictions: int
    uptime_info: str


class StatsResponse(BaseModel):
    total_predictions: int
    label_distribution: dict[str, int]
    avg_confidence: float
    recent_predictions: list[PredictResponse]


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


# ── Prediction Logic ─────────────────────────────────────────────────
def predict_sentiment(text: str) -> PredictResponse:
    """Run inference on a single text input."""
    cleaned = clean_text(text)

    if not cleaned.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is empty after preprocessing. Please provide meaningful text.",
        )

    start = time.perf_counter()
    features = VECTORIZER.transform([cleaned])
    proba = MODEL.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    elapsed_ms = (time.perf_counter() - start) * 1000

    probabilities = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}

    # Extract key word importance from TF-IDF + model coefficients
    key_words = []
    try:
        feature_names = VECTORIZER.get_feature_names_out()
        feature_indices = features.nonzero()[1]
        tfidf_scores = features.toarray()[0]
        pos_idx = 1  # positive class index
        neg_idx = 0  # negative class index
        for idx in feature_indices:
            word = feature_names[idx]
            if " " in word:
                continue
            tfidf_val = tfidf_scores[idx]
            pos_coef = MODEL.coef_[pos_idx][idx] if hasattr(MODEL, "coef_") else 0
            neg_coef = MODEL.coef_[neg_idx][idx] if hasattr(MODEL, "coef_") else 0
            net_score = float((pos_coef - neg_coef) * tfidf_val)
            word_type = "positive" if net_score > 0 else "negative"
            key_words.append({"word": word, "score": round(abs(net_score), 4), "type": word_type})
        key_words.sort(key=lambda x: x["score"], reverse=True)
        key_words = key_words[:8]
    except Exception:
        pass

    result = PredictResponse(
        text=text,
        cleaned_text=cleaned,
        prediction=LABEL_MAP[pred_idx],
        confidence=round(float(proba[pred_idx]), 4),
        probabilities=probabilities,
        inference_time_ms=round(elapsed_ms, 2),
        timestamp=datetime.now().isoformat(),
        key_words=[WordScore(**kw) for kw in key_words],
    )

    PREDICTION_LOG.append(result.model_dump())
    if len(PREDICTION_LOG) > 1000:
        PREDICTION_LOG.pop(0)

    return result


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        total_predictions=len(PREDICTION_LOG),
        uptime_info="Model is loaded and serving predictions",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict sentiment for a single text input."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_sentiment(request.text)


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predict sentiment for multiple texts."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    results = [predict_sentiment(text) for text in request.texts]
    total_ms = (time.perf_counter() - start) * 1000

    return BatchPredictResponse(
        results=results,
        total_inference_time_ms=round(total_ms, 2),
    )


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """Compare sentiment between two texts side by side."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result_a = predict_sentiment(request.text_a)
    result_b = predict_sentiment(request.text_b)

    delta = {
        label: round(result_a.probabilities[label] - result_b.probabilities[label], 4)
        for label in result_a.probabilities
    }

    return CompareResponse(
        result_a=result_a,
        result_b=result_b,
        delta=delta,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Return prediction statistics."""
    if not PREDICTION_LOG:
        return StatsResponse(
            total_predictions=0,
            label_distribution={},
            avg_confidence=0.0,
            recent_predictions=[],
        )

    labels = [p["prediction"] for p in PREDICTION_LOG]
    distribution = {label: labels.count(label) for label in set(labels)}
    avg_conf = sum(p["confidence"] for p in PREDICTION_LOG) / len(PREDICTION_LOG)

    recent = [PredictResponse(**p) for p in PREDICTION_LOG[-10:][::-1]]

    return StatsResponse(
        total_predictions=len(PREDICTION_LOG),
        label_distribution=distribution,
        avg_confidence=round(avg_conf, 4),
        recent_predictions=recent,
    )
