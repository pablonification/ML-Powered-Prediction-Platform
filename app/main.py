# main.py
"""
FastAPI application entry point.
Exposes ml_pipeline.py as HTTP REST API.

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
)
from .ml_pipeline import (
    train_model,
    predict,
    get_status,
    update_status,
    delete_model,
    load_metadata,
)


# App Initialization

app = FastAPI(
    title="Prediksi ML API",
    description="""
## Kaggle-style Machine Learning Platform API

Upload CSV data to train ML models and make predictions.

### Features
- **Model Training**: Auto-build classification/regression models from CSV
- **Prediction**: Generate predictions with trained models
- **Model Management**: List, check status, delete models

### Integration Partners
- **Jelita Frontend**: CSV upload UI
- **8EH Radio**: Listener prediction

### Developers
- **ML Pipeline**: Faiz
- **API & Deploy**: Ryota
    """,
    version="1.0.0",
    contact={"name": "Prediksi Team"},
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
        version="1.0.0"
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
            model_type=info.get("type")
        )
        for model_id, info in metadata.items()
    ]
    return ModelListResponse(models=models, count=len(models))


# Create Model (Start Training)

@app.post(
    "/training",
    response_model=TrainModelResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Training"],
    summary="Start model training",
    responses={
        202: {"description": "Training started"},
        409: {"description": "Model ID already exists", "model": ErrorResponse},
    }
)
async def create_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """
    Start async model training.
    Returns 202 Accepted immediately. Poll GET /models/{id} for status.
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
        model_type=info.get("type")
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
    tags=["Predictions"],
    summary="Make prediction",
    responses={
        404: {"description": "Model not found", "model": ErrorResponse},
        422: {"description": "Model not ready", "model": ErrorResponse},
    }
)
async def make_prediction(model_id: str, request: PredictionRequest):
    """Generate predictions using a trained model."""
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
        model_id=model_id,
        predictions=result["results"],
        count=len(result["results"])
    )


# Root

@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    """Redirect to docs."""
    return {"message": "Welcome to Prediksi ML API", "docs": "/docs", "health": "/health"}
