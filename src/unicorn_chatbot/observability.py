
"""Optional, best-effort Langfuse adapter; telemetry never affects chat responses."""
from __future__ import annotations

from typing import Any

from .config import Settings


class Observability:
    def __init__(self, settings: Settings) -> None:
        self.enabled = bool(
            settings.langfuse_enabled
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
        )
        self.capture_content = settings.langfuse_capture_content
        self.client: Any | None = None

        if self.enabled:
            try:
                from langfuse import Langfuse

                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    base_url=settings.langfuse_base_url,
                )
            except Exception as exc:
                print(f"Langfuse initialization failed: {exc}")
                self.enabled = False

    def stage(
        self,
        request_id: str,
        session_id: str,
        name: str,
        *,
        input_text: str | None = None,
        output_text: str | None = None,
        latency_ms: float | None = None,
        sources: list[dict[str, Any]] | None = None,
        **metadata: Any,
    ) -> None:
        """Record a Langfuse observation without affecting chatbot responses."""
        if not self.client:
            return

        try:
            from langfuse import propagate_attributes

            # Only include actual retrieval scores supplied by the retriever.
            score_fields = ("score", "similarity_score", "match_score", "relevance_score")
            retrieval_results = []

            for source in sources or []:
                score_key = next(
                    (key for key in score_fields if source.get(key) is not None),
                    None,
                )
                result = {
                    "rank": source.get("rank"),
                    "company": source.get("company"),
                }
                if score_key is not None:
                    result["match_score"] = source[score_key]
                    result["score_field"] = score_key

                retrieval_results.append(result)

            observation_metadata = {
                "request_id": request_id,
                "retrieval_strategy": name,
                **metadata,
            }

            if latency_ms is not None:
                observation_metadata["request_latency_ms"] = round(latency_ms, 2)

            if retrieval_results:
                observation_metadata["retrieval_results"] = retrieval_results

            observation_args: dict[str, Any] = {
                "as_type": "span",
                "name": name,
                "metadata": observation_metadata,
            }

            # Respect the content-capture privacy setting.
            if self.capture_content:
                observation_args["input"] = input_text
                observation_args["output"] = output_text

            with propagate_attributes(
                session_id=session_id,
                metadata={"request_id": request_id},
            ):
                with self.client.start_as_current_observation(
                    **observation_args
                ):
                    pass

        except Exception as exc:
            # Telemetry errors must never break the chatbot response.
            print(f"Langfuse observation failed: {exc}")

    def flush(self) -> None:
        if self.client:
            try:
                self.client.flush()
            except Exception as exc:
                print(f"Langfuse flush failed: {exc}")
