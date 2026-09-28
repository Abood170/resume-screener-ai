"""Validated, bounded requests with readiness and explicit failure responses."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
from src.predict import Predictor, InvalidResumeError
from src.extract import extract_text, FileValidationError
from api.uploads import read_cv_upload
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=1, max_length=50000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class CategoryProbability(BaseModel):
    category: str
    probability: float = Field(ge=0, le=1)


class TextStats(BaseModel):
    word_count: int = Field(ge=0)
    short_input: bool


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float = Field(ge=0, le=1)
    top_predictions: list[CategoryProbability] = Field(min_length=3, max_length=3)
    top_terms: list[str] = Field(max_length=8)
    text_stats: TextStats
    source: Literal["text", "file"]
    text_preview: str | None = Field(default=None, max_length=500)


def create_app(model_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        try:
            app.state.predictor = Predictor(model_dir)
        except Exception:
            logger.exception("Model initialization failed")
        yield
        app.state.predictor = None

    app = FastAPI(title="Resume Screener AI", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "*"
        ],  # TODO: restrict to the actual frontend origin before production deployment
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        return {"status": "ok", "model": predictor.model_name}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: ResumeRequest, request: Request) -> dict:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        try:
            return predictor.predict(payload.text)
        except InvalidResumeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Inference failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Prediction failed") from exc

    @app.post(
        "/predict/file",
        response_model=PredictionResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string", "format": "binary"}
                            },
                        }
                    }
                },
            }
        },
    )
    async def predict_file(request: Request) -> dict:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        try:
            filename, content = await read_cv_upload(request)
            text = await run_in_threadpool(extract_text, filename, content)
            result = await run_in_threadpool(predictor.predict, text)
            return {**result, "source": "file", "text_preview": text[:500]}
        except FileValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
        except InvalidResumeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except Exception as exc:
            logger.error("File prediction failed (%s)", type(exc).__name__)
            raise HTTPException(
                status_code=500, detail="File prediction failed"
            ) from None

    return app


app = create_app(Path(os.environ["MODEL_DIR"]) if "MODEL_DIR" in os.environ else None)
