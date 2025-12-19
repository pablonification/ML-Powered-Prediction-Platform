# ml_pipeline.py
import json
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib


# ==========================================================
#   STORAGE INITIALIZATION
# ==========================================================
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
META_FILE = BASE / "metadata.json"

MODEL_DIR.mkdir(exist_ok=True)

if not META_FILE.exists():
  META_FILE.write_text(json.dumps({}))


# ==========================================================
#   METADATA HELPERS (MODEL STATUS)
# ==========================================================
def load_metadata():
  return json.loads(META_FILE.read_text())

def save_metadata(meta):
  META_FILE.write_text(json.dumps(meta, indent=2))

def update_status(model_id: str, status: str):
  meta = load_metadata()
  entry = meta.get(model_id, {})
  entry["status"] = status
  entry["updated_at"] = datetime.utcnow().isoformat() + "Z"
  meta[model_id] = entry
  save_metadata(meta)

def get_status(model_id: str):
  meta = load_metadata()
  return meta.get(model_id, {"status": "not_found"})


# ==========================================================
#   MODEL TRAINING PIPELINE
# ==========================================================
def train_model(model_id: str, target_col: str, training_data: list):
  """
  Train a model based on given training data.
  Automatically selects LogisticRegression or LinearRegression.
  """

  try:
    # ------------------------------------------------------------
    # API SHOULD SET STATUS = "training" BEFORE CALLING THIS
    # e.g., update_status(model_id, "training")
    # ------------------------------------------------------------

    df = pd.DataFrame(training_data)

    if target_col not in df.columns:
      raise ValueError(f"Target column '{target_col}' not found")

    # Separate features & target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Convert X to numeric (required for regression models)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Detect problem type
    is_classification = False

    # If y is non-numeric → encode → classification
    if y.dtype == "object" or y.dtype == "bool":
      is_classification = True
    else:
      # Numeric: check unique values
      unique_vals = y.unique()
      if len(unique_vals) <= 10:
        # Usually classification if small number of classes
        is_classification = True
    
    # Prepare target
    if is_classification:
      # Encode labels for logistic regression
      le = LabelEncoder()
      y_encoded = le.fit_transform(y)
      model = LogisticRegression(max_iter=200)
      model.fit(X, y_encoded)

      # Save the encoder alongside the model
      save_path = MODEL_DIR / f"{model_id}.pkl"
      feature_cols = list(X.columns)
      joblib.dump({
        "model": model, 
        "encoder": le, 
        "type": "classification",
        "feature_cols": feature_cols,
        "target_col": target_col
      }, save_path)

    else:
      # Regression
      y_numeric = pd.to_numeric(y, errors="coerce").fillna(0)
      model = LinearRegression()
      model.fit(X, y_numeric)

      save_path = MODEL_DIR / f"{model_id}.pkl"
      feature_cols = list(X.columns)
      joblib.dump({
        "model": model, 
        "type": "regression",
        "feature_cols": feature_cols,
        "target_col": target_col
      }, save_path)

    # Update metadata with model info
    meta = load_metadata()
    meta[model_id] = {
      "status": "ready",
      "updated_at": datetime.utcnow().isoformat() + "Z",
      "type": "classification" if is_classification else "regression",
      "feature_cols": feature_cols,
      "target_col": target_col
    }
    save_metadata(meta)

    return {
      "id": model_id,
      "status": "ready",
      "created_at": datetime.utcnow().isoformat() + "Z",
      "type": "classification" if is_classification else "regression",
      "feature_cols": feature_cols,
      "target_col": target_col
    }

  except Exception as e:
    traceback.print_exc()
    update_status(model_id, "failed")  # <-- API should expose this

    return {
      "id": model_id,
      "status": "failed",
      "error": str(e)
    }


# ==========================================================
#   PREDICTION PIPELINE
# ==========================================================
def predict(model_id: str, input_data: list):
  """
  Generate predictions for given model.
  """
  status = get_status(model_id).get("status")

  if status != "ready":
    return {"error": "Model not ready", "status": status}

  model_path = MODEL_DIR / f"{model_id}.pkl"
  if not model_path.exists():
    return {"error": "Model not found"}

  bundle = joblib.load(model_path)
  model = bundle["model"]
  model_type = bundle["type"]

  df = pd.DataFrame(input_data)
  df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

  preds = model.predict(df)

  # Decode classification labels
  if model_type == "classification":
    le = bundle["encoder"]
    preds = le.inverse_transform(preds.astype(int))

  return {
    "model_id": model_id,
    "results": preds.tolist()
  }


# ==========================================================
#   DELETE MODEL
# ==========================================================
def delete_model(model_id: str):
  """
  Delete a model and its metadata.
  Allows deletion of models in any status (queued, training, ready, failed).
  """
  meta = load_metadata()
  
  # Check if model exists in metadata
  if model_id not in meta:
    return {"error": "not_found"}
  
  # Delete .pkl file if it exists (for ready/failed models)
  path = MODEL_DIR / f"{model_id}.pkl"
  if path.exists():
    path.unlink()
  
  # Remove from metadata
  meta.pop(model_id, None)
  save_metadata(meta)
  
  return {"status": "deleted"}