# Sentiment Analysis Platform

A full-stack ML model serving platform that trains a text sentiment classifier and serves predictions through a FastAPI backend with a real-time React dashboard.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        React Dashboard                         │
│         Text Input → Predictions → Charts → History            │
│                     (Port 3000)                                │
└────────────────────────┬───────────────────────────────────────┘
                         │ HTTP (JSON)
┌────────────────────────▼───────────────────────────────────────┐
│                      FastAPI Backend                            │
│    /predict  /predict/batch  /health  /stats                   │
│         TF-IDF Vectorization → Logistic Regression             │
│                     (Port 8000)                                │
└────────────────────────┬───────────────────────────────────────┘
                         │ joblib
┌────────────────────────▼───────────────────────────────────────┐
│                    Serialized Model                             │
│         sentiment_model.pkl  +  tfidf_vectorizer.pkl           │
│             Trained on Sentiment140 (1.6M tweets)              │
└────────────────────────────────────────────────────────────────┘
```

## Features

**Model Training Pipeline**
- TF-IDF vectorization with bigram support and sublinear term frequency scaling
- Multinomial Logistic Regression with 3-class output (positive, negative, neutral)
- Full evaluation suite: accuracy, F1, precision, recall, confusion matrix
- Supports training on Sentiment140 (1.6M samples) or built-in demo data

**REST API**
- `POST /predict` — Single text prediction with confidence scores + key word importance
- `POST /predict/batch` — Batch inference for up to 50 texts
- `POST /compare` — Side-by-side sentiment comparison of two texts with probability deltas
- `GET /health` — Health check with model status
- `GET /stats` — Session statistics and prediction distribution
- Auto-generated Swagger docs at `/docs`
- CORS-enabled for frontend integration

**React Dashboard**
- Real-time sentiment prediction with confidence visualization
- **Comparison Mode** — Paste two texts side by side and compare sentiment with a radar overlay chart and per-class probability deltas. Includes preset comparisons (Product vs Competitor, Before vs After, etc.)
- **Word Cloud** — Aggregated key phrase visualization across all predictions, sized by frequency and colored by sentiment polarity. Hover for occurrence counts.
- **CSV Bulk Upload** — Drag-and-drop a CSV file to analyze hundreds of texts at once with tabular results
- **Key Word Importance** — Highlights which words drove the model's prediction, adding ML explainability
- **Export to CSV/JSON** — Download your full prediction history for further analysis
- **Sample Prompts** — Pre-loaded example texts for instant demo capability
- Probability distribution bar chart and session-wide sentiment pie chart (Recharts)
- Sentiment trend line chart tracking how predictions evolve over time
- Scrollable prediction history log with clear-all functionality
- Keyboard shortcut: `Ctrl+Enter` / `⌘+Enter` to submit

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and enter the project
git clone https://github.com/YOUR_USERNAME/ml-sentiment-platform.git
cd ml-sentiment-platform

# Train the model (generates .pkl files)
pip install -r model/requirements.txt
python model/train.py --demo

# Launch the full stack
docker-compose up --build
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Option 2: Manual Setup

**Backend:**
```bash
pip install -r backend/requirements.txt
python model/train.py --demo
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

### Train on Full Dataset

Download [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) and run:

```bash
python model/train.py --data path/to/training.1600000.processed.noemoticon.csv --output model/
```

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely fantastic!"}'

# Compare two texts
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"text_a": "Best product ever!", "text_b": "Terrible, broke after one day."}'

# Response
{
  "text": "This product is absolutely fantastic!",
  "cleaned_text": "this product is absolutely fantastic",
  "prediction": "positive",
  "confidence": 0.8923,
  "probabilities": {
    "negative": 0.0412,
    "positive": 0.8923,
    "neutral": 0.0665
  },
  "inference_time_ms": 0.34,
  "timestamp": "2025-02-10T14:30:00.000000"
}
```

## Project Structure

```
ml-sentiment-platform/
├── model/
│   ├── train.py                 # Training pipeline
│   ├── requirements.txt
│   ├── sentiment_model.pkl      # Serialized model (generated)
│   ├── tfidf_vectorizer.pkl     # Serialized vectorizer (generated)
│   └── metadata.json            # Training metrics (generated)
├── backend/
│   ├── main.py                  # FastAPI application
│   └── requirements.txt
├── frontend/
│   ├── public/index.html
│   ├── src/
│   │   ├── index.js
│   │   └── App.js               # React dashboard
│   └── package.json
├── Dockerfile.backend
├── Dockerfile.frontend
├── nginx.conf
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Tech Stack

| Layer      | Technology                                              |
|------------|---------------------------------------------------------|
| ML         | scikit-learn, TF-IDF, Logistic Regression, pandas       |
| Backend    | FastAPI, Pydantic, uvicorn, joblib                      |
| Frontend   | React 18, Recharts, PapaParse, CSS-in-JS                |
| DevOps     | Docker, Docker Compose, nginx                           |

## License

MIT
