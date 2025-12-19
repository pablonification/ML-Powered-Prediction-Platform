# schemas.py
"""
Pydantic schema definitions for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any


# Training Request/Response

class TrainModelRequest(BaseModel):
    """POST /models request body."""
    id: str = Field(..., description="Unique model identifier", examples=["listener-prediction-v1"])
    target_col: str = Field(..., description="Target column name", examples=["will_churn"])
    training_data: list[dict[str, Any]] = Field(
        ...,
        description="Training data as list of dicts (CSV rows)",
        min_length=10,
        examples=[[
            {"age": 25, "hours_listened": 10, "will_churn": 0},
            {"age": 30, "hours_listened": 5, "will_churn": 1}
        ]]
    )


class TrainModelResponse(BaseModel):
    """POST /models response body."""
    id: str = Field(..., description="Model ID")
    status: str = Field(..., description="Current status", examples=["queued", "training", "ready", "failed"])
    message: str = Field(..., description="Status message")


# Model Status

class ModelStatus(BaseModel):
    """GET /models/{id} response body."""
    id: str
    status: str = Field(..., description="queued | training | ready | failed | not_found")
    updated_at: Optional[str] = Field(None, description="Last updated (ISO 8601)")
    model_type: Optional[str] = Field(None, description="classification | regression")
    feature_cols: Optional[list[str]] = Field(None, description="Feature columns used for training")
    target_col: Optional[str] = Field(None, description="Target column name")


class ModelListResponse(BaseModel):
    """GET /models response body."""
    models: list[ModelStatus]
    count: int


# Prediction Request/Response

class PredictionRequest(BaseModel):
    """POST /models/{id}/predict request body."""
    input_data: list[dict[str, Any]] = Field(
        ...,
        description="Input data for prediction (without target column)",
        min_length=1,
        examples=[[
            {"age": 28, "hours_listened": 3},
            {"age": 45, "hours_listened": 15}
        ]]
    )


class PredictionResponse(BaseModel):
    """POST /models/{id}/predict response body."""
    model_id: str
    predictions: list[Any] = Field(..., description="Prediction results", examples=[[1, 0]])
    count: int = Field(..., description="Number of predictions")


# Common Responses

class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str = Field(default="ok")
    timestamp: str
    version: str = Field(default="1.0.0")


class ErrorResponse(BaseModel):
    """Common error response format."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error details")
    detail: Optional[Any] = Field(None, description="Additional info")


class DeleteResponse(BaseModel):
    """DELETE /models/{id} response body."""
    id: str
    status: str = Field(default="deleted")
    message: str = Field(default="Model successfully deleted")
