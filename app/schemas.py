# schemas.py
"""
Pydantic schema definitions for API request/response validation.
Aligned with Predictia – 8EH Radio ITB Integrated API v1.7.0
"""

from pydantic import BaseModel, Field
from typing import Optional, Any


# ==========================================================
# FLOW 1: TRAINING
# ==========================================================

class TrainModelRequest(BaseModel):
    """POST /training request body."""
    id: str = Field(..., description="Unique model identifier", examples=["8eh-blog-engagement-v1"])
    target_col: str = Field(..., description="Target column name", examples=["readercount"])
    training_data: list[dict[str, Any]] = Field(
        ...,
        description="Training data as list of dicts. Multivalued columns (arrays, nested objects) are automatically dropped.",
        min_length=1,
        examples=[[
            {"id": "clx123abc", "title": "Mengenal Lebih Dekat 8EH Radio ITB", "category": "News", "readTime": "5 min read", "readercount": 4521},
            {"id": "clx987xyz", "title": "Top 10 Indie Bands", "category": "Music", "readTime": "8 min read", "readercount": 3105}
        ]]
    )


class TrainModelResponse(BaseModel):
    """POST /training response body (202 Accepted)."""
    id: str = Field(..., description="Model ID")
    status: str = Field(..., description="Current status", examples=["queued", "training", "ready", "failed"])
    message: str = Field(..., description="Status message")


# ==========================================================
# FLOW 2: PREDICTION
# ==========================================================

class PredictionRequest(BaseModel):
    """POST /predictions/{model_id} request body."""
    input_data: list[dict[str, Any]] = Field(
        ...,
        description="Input data for prediction (without target column)",
        min_length=1,
        examples=[[
            {"id": "clx444new", "title": "Review: Jazz Festival 2024", "category": "Music", "readTime": "7 min read"}
        ]]
    )


class PredictionResponse(BaseModel):
    """POST /predictions/{model_id} response body."""
    predictions: list[Any] = Field(..., description="Prediction results", examples=[[1850]])


# ==========================================================
# FLOW 3: SIMILARITY CHECK
# ==========================================================

class SimilarityCheckRequest(BaseModel):
    """POST /content/similarity-check request body."""
    content_1: str = Field(..., description="First content to compare", examples=["This is the first article about radio history."])
    content_2: str = Field(..., description="Second content to compare", examples=["This article discusses the origins of radio broadcasting."])


class SimilarityCheckResponse(BaseModel):
    """POST /content/similarity-check response body."""
    is_similar: bool = Field(..., description="Whether the contents are similar", examples=[True])
    similarity_level: str = Field(..., description="Level of similarity: 'identical', 'very_similar', 'similar', 'somewhat_similar', 'different'", examples=["very_similar"])
    originality_assessment: str = Field(..., description="Assessment of originality for content_1 compared to content_2", examples=["The first content appears to be an original take on the topic..."])
    detailed_analysis: str = Field(..., description="Detailed analysis of similarities and differences", examples=["Both contents discuss radio history but from different perspectives..."])


# ==========================================================
# FLOW 4: SOCIAL CAPTIONS
# ==========================================================

class SocialCaptionRequest(BaseModel):
    """POST /content/social-caption request body."""
    platform: str = Field(..., description="Target social media platform", examples=["instagram"])
    title: str = Field(..., description="Content title", examples=["Morning Show Episode 1"])
    description: str = Field(..., description="Content description", examples=["Full YouTube URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ"])


class SocialCaptionResponse(BaseModel):
    """POST /content/social-caption response body."""
    caption: str = Field(..., description="Generated social media caption", examples=["Check out the Morning Show! Link in bio. 📺"])


# ==========================================================
# FLOW 5: SUMMARIZATION
# ==========================================================

class SummarizeRequest(BaseModel):
    """POST /summarize request body."""
    content: str = Field(..., description="Content to summarize", examples=["Tune Tracker 2023-10-23. Rank 1: 'Rayuan Perempuan Gila' (stable). Rank 2: 'New Song' (up)."])


class SummarizeResponse(BaseModel):
    """POST /summarize response body."""
    summary: str = Field(..., description="Generated summary", examples=["Nadin Amizah remains at #1 this week."])


# ==========================================================
# MODEL MANAGEMENT
# ==========================================================

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


# ==========================================================
# COMMON RESPONSES
# ==========================================================

class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str = Field(default="ok")
    timestamp: str
    version: str = Field(default="1.7.0")


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
