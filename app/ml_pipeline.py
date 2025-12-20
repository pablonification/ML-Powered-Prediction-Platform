# ml_pipeline.py
"""
ML Pipeline for Predictia API.
Handles model training, prediction, and management.
Automatically drops multivalued columns (arrays, lists, nested objects) during preprocessing.
"""

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
#   DATA PREPROCESSING
# ==========================================================
def is_flat_value(value):
  """Check if a value is flat (string, number, boolean, None) - not array/dict."""
  if value is None:
    return True
  if isinstance(value, (list, dict)):
    return False
  return True


def drop_multivalued_columns(data: list[dict]) -> list[dict]:
  """
  Drop columns containing multivalued data (arrays, lists, nested objects).
  Only flat fields (strings, numbers, booleans) are preserved for training.
  
  This is required per API specification:
  - Fields like 'tags' (array) and 'authors' (nested objects) are dropped
  - Fields like 'title', 'category', 'readTime', 'readercount' are preserved
  """
  if not data:
    return data
  
  # Identify columns to drop by checking all rows
  columns_to_drop = set()
  
  for row in data:
    for key, value in row.items():
      if not is_flat_value(value):
        columns_to_drop.add(key)
  
  # Remove multivalued columns from each row
  cleaned_data = []
  for row in data:
    cleaned_row = {k: v for k, v in row.items() if k not in columns_to_drop}
    cleaned_data.append(cleaned_row)
  
  if columns_to_drop:
    print(f"[Preprocessing] Dropped multivalued columns: {columns_to_drop}")
  
  return cleaned_data


def preprocess_features(df: pd.DataFrame, feature_cols: list = None) -> tuple[pd.DataFrame, list, dict]:
  """
  Preprocess feature DataFrame for ML models.
  Handles both numeric and categorical (string) columns.
  
  Returns:
    - Processed DataFrame with numeric values
    - List of feature columns used
    - Dictionary of label encoders for categorical columns
  """
  encoders = {}
  processed_df = df.copy()
  
  for col in processed_df.columns:
    # Check if column is numeric
    if pd.api.types.is_numeric_dtype(processed_df[col]):
      # Fill NaN with 0 for numeric columns
      processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce").fillna(0)
    else:
      # Encode categorical/string columns
      le = LabelEncoder()
      # Handle NaN by converting to string first
      processed_df[col] = processed_df[col].fillna("__MISSING__").astype(str)
      processed_df[col] = le.fit_transform(processed_df[col])
      encoders[col] = le
  
  return processed_df, list(processed_df.columns), encoders


# ==========================================================
#   MODEL TRAINING PIPELINE
# ==========================================================
def train_model(model_id: str, target_col: str, training_data: list):
  """
  Train a model based on given training data.
  Automatically:
  1. Drops multivalued columns (arrays, nested objects)
  2. Encodes categorical features
  3. Selects LogisticRegression or LinearRegression based on target
  """

  try:
    # ------------------------------------------------------------
    # Step 1: Drop multivalued columns (arrays, lists, nested objects)
    # ------------------------------------------------------------
    cleaned_data = drop_multivalued_columns(training_data)
    
    df = pd.DataFrame(cleaned_data)

    if target_col not in df.columns:
      raise ValueError(f"Target column '{target_col}' not found")

    # Separate features & target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # ------------------------------------------------------------
    # Step 2: Preprocess features (handle both numeric and categorical)
    # ------------------------------------------------------------
    X_processed, feature_cols, feature_encoders = preprocess_features(X)

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
      model.fit(X_processed, y_encoded)

      # Save the encoder alongside the model
      save_path = MODEL_DIR / f"{model_id}.pkl"
      joblib.dump({
        "model": model, 
        "encoder": le,
        "feature_encoders": feature_encoders,
        "type": "classification",
        "feature_cols": feature_cols,
        "target_col": target_col
      }, save_path)

    else:
      # Regression
      y_numeric = pd.to_numeric(y, errors="coerce").fillna(0)
      model = LinearRegression()
      model.fit(X_processed, y_numeric)

      save_path = MODEL_DIR / f"{model_id}.pkl"
      joblib.dump({
        "model": model,
        "feature_encoders": feature_encoders,
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
    update_status(model_id, "failed")

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
  Automatically:
  1. Drops multivalued columns from input
  2. Applies same feature encoding as training
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
  feature_cols = bundle.get("feature_cols", [])
  feature_encoders = bundle.get("feature_encoders", {})

  # Drop multivalued columns from input data
  cleaned_data = drop_multivalued_columns(input_data)
  
  df = pd.DataFrame(cleaned_data)
  
  # Ensure we only use the feature columns the model was trained on
  for col in feature_cols:
    if col not in df.columns:
      df[col] = 0  # Add missing columns with default value
  
  # Keep only the feature columns in the correct order
  df = df[feature_cols]
  
  # Apply feature encoding
  for col in df.columns:
    if col in feature_encoders:
      le = feature_encoders[col]
      # Handle unseen categories by mapping to a default
      df[col] = df[col].fillna("__MISSING__").astype(str)
      # Transform with handling for unseen labels
      df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    else:
      df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

  preds = model.predict(df)

  # Decode classification labels
  if model_type == "classification":
    le = bundle["encoder"]
    preds = le.inverse_transform(preds.astype(int))

  return {
    "model_id": model_id,
    "predictions": preds.tolist()
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