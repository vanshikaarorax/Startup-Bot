"""Optional Langfuse adapter; telemetry never affects chat responses."""
from __future__ import annotations

from typing import Any

from ..core.config import Settings


class Observability:
    def __init__(self, settings: Settings) -> None:
        self.enabled = bool(settings.langfuse_enabled and settings.langfuse_public_key and settings.langfuse_secret_key)
        self.capture_content = settings.langfuse_capture_content
        self.client: Any | None = None
        if self.enabled:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key, base_url=settings.langfuse_base_url)
            except Exception:
                self.enabled = False

    def stage(self, request_id: str, session_id: str, name: str, *, input_text: str | None = None, output_text: str | None = None, latency_ms: float | None = None, sources: list[dict[str, Any]] | None = None, **metadata: Any) -> None:
        if not self.client:
            return
        try:
            from langfuse import propagate_attributes
            score_keys = ("score", "similarity_score", "match_score", "relevance_score")
            results = []
            for source in sources or []:
                key = next((item for item in score_keys if source.get(item) is not None), None)
                result = {"rank": source.get("rank"), "company": source.get("company")}
                if key: result.update(match_score=source[key], score_field=key)
                results.append(result)
            observation_metadata = {"request_id": request_id, "retrieval_strategy": name, **metadata}
            if latency_ms is not None: observation_metadata["request_latency_ms"] = round(latency_ms, 2)
            if results: observation_metadata["retrieval_results"] = results
            arguments: dict[str, Any] = {"as_type": "span", "name": name, "metadata": observation_metadata}
            if self.capture_content: arguments.update(input=input_text, output=output_text)
            with propagate_attributes(session_id=session_id, metadata={"request_id": request_id}):
                with self.client.start_as_current_observation(**arguments):
                    pass
        except Exception:
            pass

    def flush(self) -> None:
        if self.client:
            try: self.client.flush()
            except Exception: pass
