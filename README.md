---
title: Predictia ML API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# ML-Powered Prediction Platform

Kaggle-style Machine Learning Platform API for II3160 Integrated System Technology course.

## Overview

This platform allows users to upload CSV data, train ML models, and make predictions via REST API.

## Features

- **Model Training**: Auto-build classification/regression models from CSV data
- **Prediction**: Generate predictions with trained models  
- **Model Management**: List, check status, delete models
- **Lightweight**: Uses Logistic/Linear Regression optimized for STB deployment

## Tech Stack

- **Framework**: FastAPI
- **ML**: scikit-learn (Logistic Regression, Linear Regression)
- **Storage**: Local filesystem (models/ + metadata.json)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List all models |
| POST | `/training` | Start model training |
| GET | `/models/{id}` | Get model status |
| DELETE | `/models/{id}/delete` | Delete a model |
| POST | `/predictions/{id}` | Make predictions |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ML-Powered-Prediction-Platform.git
cd ML-Powered-Prediction-Platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access API Documentation

Open http://localhost:8000/docs for Swagger UI.

## Usage Examples

### Train a Model

```bash
curl -X POST http://localhost:8000/training \
  -H "Content-Type: application/json" \
  -d '{
    "id": "churn-predictor",
    "target_col": "will_churn",
    "training_data": [
      {"age": 25, "hours": 10, "will_churn": 0},
      {"age": 30, "hours": 5, "will_churn": 1}
    ]
  }'
```

### Make Predictions

```bash
curl -X POST http://localhost:8000/predictions/churn-predictor \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": [
      {"age": 28, "hours": 3}
    ]
  }'
```

## Project Structure

```
prediksi-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI entry point
│   ├── ml_pipeline.py   # ML training & prediction logic
│   ├── schemas.py       # Pydantic models
│   └── models/          # Saved model files (.pkl)
├── requirements.txt
├── .gitignore
└── README.md
```