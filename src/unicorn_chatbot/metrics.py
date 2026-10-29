"""Process-local, concurrency-safe operational counters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock


@dataclass
class MetricSnapshot:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    clarification_questions: int = 0
    fallback_responses: int = 0
    retrieval_no_matches: int = 0
    llm_call_failures: int = 0
    total_latency_ms: float = 0.0

    def values(self) -> dict[str, float | int]:
        output = asdict(self)
        output["average_latency_ms"] = self.total_latency_ms / self.total_requests if self.total_requests else 0.0
        output["error_rate"] = self.failed_requests / self.total_requests if self.total_requests else 0.0
        output["clarification_rate"] = self.clarification_questions / self.total_requests if self.total_requests else 0.0
        return output


class MetricsService:
    def __init__(self) -> None:
        self._snapshot = MetricSnapshot()
        self._lock = Lock()

    def record(self, *, outcome: str, latency_ms: float, llm_failure: bool = False) -> None:
        with self._lock:
            item = self._snapshot
            item.total_requests += 1
            item.total_latency_ms += latency_ms
            if outcome == "failed": item.failed_requests += 1
            else: item.successful_requests += 1  # clarification is a successful response.
            if outcome == "clarification": item.clarification_questions += 1
            if outcome == "fallback": item.fallback_responses += 1
            if outcome == "no_match": item.retrieval_no_matches += 1
            if llm_failure: item.llm_call_failures += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return self._snapshot.values()
