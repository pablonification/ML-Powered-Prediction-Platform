# main.py
"""
Predictia – 8EH Radio ITB Integrated API
FastAPI application entry point.

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from .schemas import (
    TrainModelRequest,
    TrainModelResponse,
    ModelStatus,
    ModelListResponse,
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
    DeleteResponse,
    SimilarityCheckRequest,
    SimilarityCheckResponse,
    SocialCaptionRequest,
    SocialCaptionResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from .ml_pipeline import (
    train_model,
    predict,
    get_status,
    update_status,
    delete_model,
    load_metadata,
)
from .content_service import (
    check_similarity,
    generate_social_caption,
    summarize_text,
)


# App Initialization

app = FastAPI(
    title="Predictia – 8EH Radio ITB Integrated API",
    description="""
## Integrated API Documentation

**Integration Strategy:**
* **Predictia (AI):** Agnostic endpoints for ML training and prediction.
* **8EH Radio (Content):** Content services with Gemini AI.

**Specific Logic:**
* **Training:** Uses POST /training. Multivalued columns are dropped.
* **Prediction:** Uses POST /predictions/{model_id}.

### Features
- **Model Training**: Auto-build classification/regression models from data
- **Prediction**: Generate predictions with trained models
- **Content Similarity**: Check content against existing corpus
- **Social Captions**: Generate platform-specific social media captions
- **Summarization**: AI-powered text summarization

### Integration Partners
- **8EH Radio ITB**: Content management and prediction
    """,
    version="1.7.0",
    contact={"name": "Predictia Team"},
)


# CORS Middleware
# Allows cross-origin requests from frontend and partner services

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handler

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch unexpected errors and return unified JSON format."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "message": str(exc), "detail": None}
    )


# Health Check

@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
async def health_check():
    """Check if API is running."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.7.0"
    )


# List Models

@app.get("/models", response_model=ModelListResponse, tags=["Models"], summary="List all models")
async def list_models():
    """Get list of all registered models and their status."""
    metadata = load_metadata()
    models = [
        ModelStatus(
            id=model_id,
            status=info.get("status", "unknown"),
            updated_at=info.get("updated_at"),
            model_type=info.get("type"),
            feature_cols=info.get("feature_cols"),
            target_col=info.get("target_col")
        )
        for model_id, info in metadata.items()
    ]
    return ModelListResponse(models=models, count=len(models))


# Create Model (Start Training)

@app.post(
    "/training",
    response_model=TrainModelResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Predictia"],
    summary="Train AI Model",
    responses={
        202: {"description": "Training queued"},
        409: {"description": "Model ID already exists", "model": ErrorResponse},
    }
)
async def create_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """
    Accepts any dataset to train the model.
    
    **Data Processing Note:** Multivalued columns (arrays, lists, nested objects) 
    will be automatically **dropped** during preprocessing. Only flat fields 
    (strings, numbers, booleans) are used for training.
    """
    model_id = request.id

    # Check for duplicate ID
    existing = get_status(model_id)
    if existing.get("status") != "not_found":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "conflict",
                "message": f"Model '{model_id}' already exists",
                "current_status": existing.get("status")
            }
        )

    # Queue training
    update_status(model_id, "queued")
    background_tasks.add_task(run_training, model_id, request.target_col, request.training_data)

    return TrainModelResponse(
        id=model_id,
        status="queued",
        message="Training job accepted. Check GET /models/{id} for status."
    )


def run_training(model_id: str, target_col: str, training_data: list):
    """Background task: run model training."""
    update_status(model_id, "training")
    train_model(model_id, target_col, training_data)


# Get Model Status

@app.get(
    "/models/{model_id}",
    response_model=ModelStatus,
    tags=["Models"],
    summary="Get model status",
    responses={404: {"description": "Model not found", "model": ErrorResponse}}
)
async def get_model_status(model_id: str):
    """Get current status of a model."""
    info = get_status(model_id)

    if info.get("status") == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Model '{model_id}' not found"}
        )

    return ModelStatus(
        id=model_id,
        status=info.get("status"),
        updated_at=info.get("updated_at"),
        model_type=info.get("type"),
        feature_cols=info.get("feature_cols"),
        target_col=info.get("target_col")
    )


# Delete Model

@app.delete(
    "/models/{model_id}/delete",
    response_model=DeleteResponse,
    tags=["Models"],
    summary="Delete model",
    responses={404: {"description": "Model not found", "model": ErrorResponse}}
)
async def remove_model(model_id: str):
    """Delete a model and its metadata."""
    result = delete_model(model_id)

    if result.get("error") == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Model '{model_id}' not found"}
        )

    return DeleteResponse(id=model_id, status="deleted", message="Model successfully deleted")


# Make Prediction

@app.post(
    "/predictions/{model_id}",
    response_model=PredictionResponse,
    tags=["Predictia"],
    summary="Make Prediction",
    responses={
        404: {"description": "Model not found", "model": ErrorResponse},
        422: {"description": "Model not ready", "model": ErrorResponse},
    }
)
async def make_prediction(model_id: str, request: PredictionRequest):
    """
    Predicts target variable using a trained model.
    Multivalued columns in input_data are automatically dropped.
    """
    info = get_status(model_id)
    current_status = info.get("status")

    if current_status == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Model '{model_id}' not found"}
        )

    if current_status != "ready":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "model_not_ready",
                "message": f"Model '{model_id}' is not ready",
                "current_status": current_status
            }
        )

    result = predict(model_id, request.input_data)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "prediction_failed", "message": result["error"]}
        )

    return PredictionResponse(
        predictions=result["predictions"]
    )


# ==========================================================
# CONTENT SERVICES (Gemini AI)
# ==========================================================

# Similarity Check

@app.post(
    "/content/similarity-check",
    response_model=SimilarityCheckResponse,
    tags=["Predictia"],
    summary="Check Content Similarity",
    responses={
        500: {"description": "Similarity check failed", "model": ErrorResponse},
    }
)
async def similarity_check(request: SimilarityCheckRequest):
    """
    Check similarity between two provided contents.
    Uses Gemini AI to analyze semantic similarity and originality.
    Returns detailed analysis including similarity level, originality assessment, and explanation.
    """
    try:
        result = await check_similarity(request.content_1, request.content_2)
        return SimilarityCheckResponse(
            is_similar=result["is_similar"],
            similarity_level=result["similarity_level"],
            originality_assessment=result["originality_assessment"],
            detailed_analysis=result["detailed_analysis"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "similarity_check_failed", "message": str(e)}
        )


# Social Caption Generation

@app.post(
    "/content/social-caption",
    response_model=SocialCaptionResponse,
    tags=["Predictia"],
    summary="Generate Caption",
    responses={
        500: {"description": "Caption generation failed", "model": ErrorResponse},
    }
)
async def social_caption(request: SocialCaptionRequest):
    """
    Generate social media caption for content.
    Uses Gemini AI to create platform-specific engaging captions.
    """
    try:
        result = await generate_social_caption(
            request.platform, 
            request.title, 
            request.description
        )
        return SocialCaptionResponse(caption=result["caption"])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "caption_generation_failed", "message": str(e)}
        )


# Text Summarization

@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    tags=["Predictia"],
    summary="Summarize Text",
    responses={
        500: {"description": "Summarization failed", "model": ErrorResponse},
    }
)
async def summarize(request: SummarizeRequest):
    """
    Summarize text content.
    Uses Gemini AI to generate concise summaries.
    """
    try:
        result = await summarize_text(request.content)
        return SummarizeResponse(summary=result["summary"])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "summarization_failed", "message": str(e)}
        )


# Root

@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    """Redirect to docs."""
    return {"message": "Welcome to Predictia API", "docs": "/docs", "health": "/health", "version": "1.7.0"}
