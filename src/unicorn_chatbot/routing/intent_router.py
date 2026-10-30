"""LLM-free routing for independently answerable dataset requests."""
from __future__ import annotations

import re

from ..data import UnicornRepository, normalize_location_text, normalize_search_text
from .schemas import RouteDecision


class IntentRouter:
    """Recognizes only exact, schema-backed requests; all else is conversational."""

    _CONTEXT = re.compile(r"\b(it|its|they|their|them|these|those|which of (?:these|them)|first|second|third|compare|same one|what about)\b", re.I)
    _YEAR = re.compile(r"\b(?:founded|started|established)\s+(?:(after|since|before|until|on|in)\s+)?(19\d{2}|20\d{2})\b", re.I)

    def __init__(self, repository: UnicornRepository) -> None:
        self.repository = repository

    @staticmethod
    def _mentioned(message: str, values: list[str], *, location: bool = False) -> str | None:
        normalizer = normalize_location_text if location else normalize_search_text
        text = normalizer(message)
        matches = [value for value in values if normalizer(value) and normalizer(value) in text]
        return max(matches, key=lambda value: len(normalizer(value))) if matches else None

    def route(self, message: str) -> RouteDecision:
        # Explicit named entities are independent requests even in a session.
        companies = tuple(self.repository.mentioned_companies(message))
        if companies:
            return RouteDecision("fast", "company_lookup", company_names=companies, reason="explicit_company")
        filters: dict[str, str | int] = {}
        for kind, location in (("sector", False), ("location", True), ("country", False), ("stage", False)):
            value = self._mentioned(message, self.repository.values_for(kind), location=location)
            if value:
                filters[kind] = value
        year = self._YEAR.search(message)
        if year:
            operator = {"after": "gt", "since": "gte", "before": "lt", "until": "lte"}.get((year.group(1) or "").casefold(), "eq")
            filters["founded_year"] = int(year.group(2))
            filters["founded_year_operator"] = operator
        if self._CONTEXT.search(message):
            return RouteDecision("conversation", "contextual", filters=filters, reason="context_reference")
        if filters:
            return RouteDecision("fast", "structured_filter", filters=filters, reason="schema_backed_filter")
        return RouteDecision("conversation", "unresolved", reason="semantic_or_ambiguous")
