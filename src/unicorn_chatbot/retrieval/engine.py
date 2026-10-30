"""Single retrieval priority point for all chatbot paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..routing import RouteDecision
from .company_lookup import CompanyLookup
from .contextual_retriever import ContextualRetriever
from .semantic_retriever import SemanticRetriever
from .structured_retriever import StructuredRetriever


@dataclass(slots=True)
class RetrievalResult:
    records: list[dict[str, Any]]
    strategy: str
    scored_results: list[dict[str, Any]]


class RetrievalEngine:
    def __init__(self, company_lookup: CompanyLookup, structured: StructuredRetriever, contextual: ContextualRetriever, semantic: SemanticRetriever | None, min_score: float) -> None:
        self.company_lookup, self.structured, self.contextual = company_lookup, structured, contextual
        self.semantic, self.min_score = semantic, min_score

    def retrieve(self, query: str, decision: RouteDecision, *, previous_records: list[dict[str, Any]] | None = None, use_previous_results: bool = False) -> RetrievalResult:
        if decision.company_names:
            records = self.company_lookup.retrieve(decision.company_names)
            if records: return RetrievalResult(records, "company_lookup", [])
        if use_previous_results and previous_records:
            records = self.contextual.retrieve(query, previous_records, decision.filters)
            strategy = "context_company_lookup" if len(records) == 1 else "context_filter" if decision.filters else "context_results"
            return RetrievalResult(records, strategy, [])
        if decision.filters:
            records = self.structured.retrieve(decision.filters)
            return RetrievalResult(records, "structured", [])
        if self.semantic is None:
            return RetrievalResult([], "unavailable", [])
        scored = self.semantic.search(query, min_score=self.min_score)
        return RetrievalResult([item["record"] for item in scored], "semantic", scored)
