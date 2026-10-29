
"""FastAPI surface with validated sessions, safe failures, logs, and metrics."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .config import Settings
from .data import DatasetError, UnicornRepository
from .logging_utils import JsonEventLogger
from .metrics import MetricsService
from .observability import Observability
from .retrieval import SemanticRetriever
from .sanitization import InputError, sanitize_message, validate_session_id
from .service import ChatbotService


class ChatRequest(BaseModel):
    message: str = Field(max_length=10000)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    retrieval_strategy: str
    session_id: str


def create_app(
    settings: Settings | None = None,
    service: ChatbotService | None = None,
    metrics: MetricsService | None = None,
    logger: JsonEventLogger | None = None,
    observability: Observability | None = None,
) -> FastAPI:
    configured = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.metrics = metrics or MetricsService()
        app.state.logger = logger or JsonEventLogger()
        app.state.observability = observability or Observability(configured)

        if service is not None:
            app.state.service = service
        else:
            try:
                repo = UnicornRepository.from_csv(configured.dataset_path)
                app.state.service = ChatbotService(
                    repo,
                    configured,
                    SemanticRetriever(
                        repo,
                        configured.embedding_model,
                        configured.vector_index_path,
                    ),
                )
            except DatasetError as exc:
                app.state.startup_error = str(exc)

        yield

        app.state.observability.flush()

    app = FastAPI(
        title="Indian Unicorn Chatbot",
        version="0.2.0",
        lifespan=lifespan,
    )

    def get_service(request: Request) -> ChatbotService:
        if hasattr(request.app.state, "startup_error"):
            raise HTTPException(
                503,
                request.app.state.startup_error,
            )
        return request.app.state.service

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        if hasattr(request.app.state, "startup_error"):
            return {
                "status": "degraded",
                "detail": request.app.state.startup_error,
            }
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics_endpoint(request: Request) -> dict[str, float | int]:
        return request.app.state.metrics.snapshot()

    @app.delete("/sessions/{session_id}", status_code=204)
    def clear_session(
        session_id: str,
        chatbot: ChatbotService = Depends(get_service),
    ) -> None:
        try:
            chatbot.clear_session(validate_session_id(session_id))
        except InputError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        request: Request,
        chatbot: ChatbotService = Depends(get_service),
    ) -> ChatResponse:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()

        try:
            message = sanitize_message(
                payload.message,
                configured.max_message_length,
            )
            session_id = validate_session_id(payload.session_id)

            answer, sources, strategy = chatbot.answer(message, session_id)

            # Measure actual chatbot request time before telemetry overhead.
            latency_ms = (time.perf_counter() - started) * 1000

            outcome = (
                "clarification"
                if strategy == "clarification"
                else "no_match"
                if not sources
                else "fallback"
                if strategy == "fallback"
                else "success"
            )

            request.app.state.metrics.record(
                outcome=outcome,
                latency_ms=latency_ms,
            )

            request.app.state.logger.event(
                "chat_completed",
                request_id=request_id,
                session_id=session_id,
                outcome=outcome,
                retrieval_count=len(sources),
                clarification=strategy == "clarification",
                model=configured.openai_model,
            )

            request.app.state.observability.stage(
                request_id,
                session_id,
                strategy,
                input_text=message,
                output_text=answer,
                latency_ms=latency_ms,
                sources=sources,
                retrieval_count=len(sources),
                model=configured.openai_model,
                outcome=outcome,
            )

            return ChatResponse(
                answer=answer,
                sources=sources,
                retrieval_strategy=strategy,
                session_id=session_id,
            )

        except InputError as exc:
            request.app.state.metrics.record(
                outcome="failed",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            request.app.state.logger.event(
                "chat_failed",
                request_id=request_id,
                error_category="validation",
            )
            raise HTTPException(422, str(exc)) from exc

        except Exception:
            request.app.state.metrics.record(
                outcome="failed",
                latency_ms=(time.perf_counter() - started) * 1000,
                llm_failure=True,
            )
            request.app.state.logger.event(
                "chat_failed",
                request_id=request_id,
                error_category="internal",
            )
            raise HTTPException(
                500,
                "Unable to process the request.",
            )

    return app


app = create_app()
