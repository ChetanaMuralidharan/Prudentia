from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model_service import ModelService
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    LoanApplication,
    ModelInfoResponse,
    PredictionResponse,
)


logger = logging.getLogger("prudentia_api")
service: ModelService | None = None
startup_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service, startup_error
    try:
        service = ModelService()
        startup_error = None
        logger.info("Prudentia model service loaded.")
    except Exception as exc:
        service = None
        startup_error = str(exc)
        logger.exception("Failed to load model service.")
    yield


app = FastAPI(
    title="Prudentia Credit Risk API",
    version="1.0.0",
    description="Calibrated credit-risk scoring, decisioning, risk tiers and reason codes.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> ModelService:
    if service is None:
        raise HTTPException(status_code=503, detail=f"Model service unavailable: {startup_error}")
    return service


@app.get("/")
def root() -> dict:
    return {"name": "Prudentia Credit Risk API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    current = get_service()
    return HealthResponse(
        status="healthy",
        model_loaded=current.calibrated_model is not None,
        preprocessor_loaded=current.preprocessor is not None,
        defaults_loaded=bool(current.defaults),
        decision_threshold=current.threshold,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**get_service().model_info())


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> PredictionResponse:
    try:
        result = get_service().predict_one(application.model_dump(exclude_none=True))
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if not request.applications:
        raise HTTPException(status_code=400, detail="At least one application is required.")
    if len(request.applications) > 1000:
        raise HTTPException(status_code=400, detail="Maximum batch size is 1,000 applications.")
    try:
        rows = [application.model_dump(exclude_none=True) for application in request.applications]
        predictions = get_service().predict_many(rows)
        return BatchPredictionResponse(
            prediction_count=len(predictions),
            predictions=[PredictionResponse(**prediction) for prediction in predictions],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch prediction failed.")
        raise HTTPException(status_code=422, detail=f"Batch prediction failed: {exc}") from exc
