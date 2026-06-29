# MediPredict AI — Symptom-to-Disease Predictor

> ⚠️ **Medical Disclaimer**: This is an **educational/portfolio project** and does **not** constitute medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional for any health concerns.

---

## Overview

An AI-powered web application that predicts the most likely disease(s) from a set of user-provided symptoms. Built with Python, scikit-learn, and Flask.

**Features:**
- 🔍 **Checklist mode** — searchable grid of 100+ symptoms
- 💬 **Free-text mode** — type symptoms in plain English, matched via NLP (synonyms + fuzzy matching)
- 📊 **Top-3 predictions** with confidence scores and an interactive bar chart
- 📋 **Disease info** — description, home care precautions, severity, "see a doctor" flag
- 🚨 **Red-flag detection** — alerts for emergency symptoms like chest pain or breathlessness
- 🎨 **Premium dark UI** — glassmorphism, animated gradient, Inter font

---

## Dataset

| Property | Value |
|---|---|
| Source | Based on [Kaggle Disease Prediction Using ML](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) |
| Diseases | 41 |
| Symptoms (features) | ~100+ binary features |
| Training samples | ~4,920 |
| Generation | `data/generate_dataset.py` (self-contained, no download needed) |

---

## Model Performance

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| **Random Forest** ✅ | ~99% | ~99% |
| Decision Tree | ~97% | ~97% |
| Naive Bayes (Bernoulli) | ~89% | ~89% |

→ **Random Forest** (200 estimators) is selected as the production model.

After training, full metrics including a confusion matrix are saved to `model/metrics.json`.

---

## Project Structure

```
ai disease predictor/
├── data/
│   ├── generate_dataset.py    # Generates training.csv and testing.csv
│   ├── training.csv           # Generated training data
│   ├── testing.csv            # Generated test data
│   └── symptom_descriptions.json  # Disease info, precautions, severity
│
├── model/
│   ├── train.py               # Trains 3 models, saves best + artifacts
│   ├── best_model.pkl         # Saved best model (Random Forest)
│   ├── label_encoder.pkl      # Disease label encoder
│   ├── features.pkl           # Feature column names list
│   └── metrics.json           # Accuracy, F1, confusion matrix
│
├── static/
│   ├── css/style.css          # Premium dark design system
│   └── js/app.js              # Frontend logic + Chart.js rendering
│
├── templates/
│   └── index.html             # Flask Jinja2 template
│
├── app.py                     # Flask backend (REST API + serving)
├── utils.py                   # Preprocessing, NLP matching, disease lookup
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_dataset.py
```

### 3. Train the model
```bash
python model/train.py
```

This will print model comparison results and save the best model to `model/`.

### 4. Start the Flask server
```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main web UI |
| `GET` | `/symptoms` | Returns JSON list of all known symptoms |
| `POST` | `/predict` | Predict diseases from symptoms |
| `GET` | `/health` | Health check |

### POST `/predict` — Checklist mode
```json
{ "symptoms": ["high_fever", "headache", "chills", "vomiting"] }
```

### POST `/predict` — Free text mode
```json
{ "text": "I have a bad headache, fever, and I feel nauseous" }
```

### Response
```json
{
  "predictions": [
    {
      "disease": "Malaria",
      "confidence": 0.85,
      "confidence_pct": "85.0%",
      "description": "...",
      "precautions": ["...", "..."],
      "see_doctor": true,
      "severity": "severe"
    }
  ],
  "symptoms_used": ["High Fever", "Headache", "Chills"],
  "red_flags": [],
  "is_emergency": false,
  "disclaimer": "This prediction is for educational purposes only..."
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML / Training | Python, scikit-learn, pandas, numpy |
| Backend | Flask 3.0 |
| Frontend | Vanilla HTML5, CSS3, JavaScript |
| Charts | Chart.js 4.4 |
| Fonts | Inter (Google Fonts) |
| Model Persistence | joblib |

---

## ⚠️ Disclaimer (Repeated Intentionally)

This tool is an **educational and portfolio project**. It is **not a medical device**, not FDA-approved, and should **never** be used as a substitute for professional medical advice, diagnosis, or treatment. The predictions are based on statistical patterns in training data and may be incorrect.

**If you are experiencing a medical emergency, call 911 (or your local emergency number) immediately.**
