
"""Safe session-aware orchestration over deterministic and semantic retrieval."""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from .clarification import ClarificationPolicy
from .config import Settings
from .data import (
    UnicornRepository,
    normalize_location_text,
    normalize_search_text,
)
from .prompts import build_chat_prompt
from .retrieval import SemanticRetriever
from .sessions import SessionManager

logger = logging.getLogger(__name__)


class ChatbotService:
    _ENTITY_REFERENCE = re.compile(
        r"\b(?:it|its|they|their|them|these|those|"
        r"which\s+of\s+them|which\s+of\s+these|"
        r"that\s+company|the\s+company|the\s+results|"
        r"that\s+result|those\s+companies|remaining\s+companies|"
        r"the\s+first\s+one|the\s+second\s+one)\b",
        re.IGNORECASE,
    )

    _ATTRIBUTE_FOLLOWUP = re.compile(
        r"\b(?:valuation|revenue|funding|investors?|employees?|"
        r"employee\s+count|founded|founded\s+year|location|based|"
        r"sector|industry|stage|round|highest|lowest|summari[sz]e)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        repository: UnicornRepository,
        settings: Settings,
        retriever: SemanticRetriever | None = None,
        sessions: SessionManager | None = None,
        clarification: ClarificationPolicy | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.retriever = retriever
        self.sessions = sessions or SessionManager(settings.max_session_turns)
        self.clarification = clarification or ClarificationPolicy()

    def _extract_filters(
        self, message: str
    ) -> tuple[str | None, str | None]:
        def mentioned(column: str | None, kind: str) -> str | None:
            if not column:
                return None

            normalizer = (
                normalize_location_text
                if kind == "location"
                else normalize_search_text
            )
            text = normalizer(message)

            for value in (
                self.repository.frame[column]
                .dropna()
                .astype(str)
                .unique()
            ):
                if normalizer(value) and normalizer(value) in text:
                    return value

            # Handle the common Bangalore/Bengaluru naming difference.
            if kind == "location" and re.search(
                r"\bbangalore\b", message, re.IGNORECASE
            ):
                for value in (
                    self.repository.frame[column]
                    .dropna()
                    .astype(str)
                    .unique()
                ):
                    if "bengaluru" in normalizer(value):
                        return value

            return None

        return (
            mentioned(self.repository._column_for("sector"), "sector"),
            mentioned(self.repository._column_for("location"), "location"),
        )

    @staticmethod
    def _select_context_records(
        message: str, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Resolve ordinal references against the previous result set."""
        text = message.casefold()

        if re.search(r"\bfirst\s+two\b", text):
            return records[:2]
        if re.search(r"\bsecond\b", text):
            return records[1:2]
        if re.search(r"\bfirst\b", text):
            return records[:1]

        return records

    def _resolve_context(
        self,
        message: str,
        records: list[dict[str, Any]],
        sector: str | None,
        location: str | None,
    ) -> tuple[list[dict[str, Any]], str] | None:
        """Resolve conversational follow-ups using the previous evidence."""
        if not records:
            return None

        reference = bool(self._ENTITY_REFERENCE.search(message))
        attribute_followup = bool(self._ATTRIBUTE_FOLLOWUP.search(message))

        if not (reference or attribute_followup):
            return None

        selected = self._select_context_records(message, records)

        if sector or location:
            for kind, value in (
                ("sector", sector),
                ("location", location),
            ):
                column = self.repository._column_for(kind)
                if not value or not column:
                    continue

                normalizer = (
                    normalize_location_text
                    if kind == "location"
                    else normalize_search_text
                )
                target = normalizer(value)

                selected = [
                    record
                    for record in selected
                    if target
                    in normalizer(str(record.get(column, "")))
                ]

            return selected, "context_filter"

        strategy = (
            "context_company_lookup"
            if len(selected) == 1
            else "context_results"
        )
        return selected, strategy

    @staticmethod
    def _safety_response(message: str) -> str | None:
        text = message.casefold()

        if any(
            phrase in text
            for phrase in ("api key", "system prompt", "reveal your")
        ):
            return (
                "I can’t provide credentials, internal instructions, "
                "or configuration. I can help with questions supported "
                "by the dataset."
            )

        if "use the internet" in text or "web search" in text:
            return (
                "I can only use the supplied dataset and cannot use "
                "live web information."
            )

        if "invent" in text and (
            "revenue" in text or "dataset" in text
        ):
            return (
                "I can’t invent missing company information. I can answer "
                "only from the supplied dataset fields."
            )

        return None

    def retrieve(
        self, message: str, session_id: str = "default"
    ) -> tuple[list[dict[str, Any]], str]:
        state = self.sessions.get(session_id)
        sector, location = self._extract_filters(message)

        # Prefer explicit company-name matches.
        lowered_message = message.casefold()
        names = [
            name
            for name in (
                self.repository.frame[self.repository.name_column]
                .dropna()
                .astype(str)
                .unique()
            )
            if name.casefold() in lowered_message
        ]

        if names:
            records = [
                record
                for name in names
                for record in self.repository.find_companies(name)
            ]
            return records, "company_lookup"

        # Resolve follow-ups against this session's previous results.
        context = self._resolve_context(
            message, state.records, sector, location
        )
        if context is not None:
            return context

        # For a new, non-contextual query, use deterministic filters.
        if sector or location:
            return (
                self.repository.filter(sector=sector, location=location),
                "structured",
            )

        if not self.retriever:
            return [], "unavailable"

        results = self.retriever.search(
            message,
            min_score=self.settings.retrieval_min_score,
        )
        return [item["record"] for item in results], "semantic"

    def answer(
        self, message: str, session_id: str = "default"
    ) -> tuple[str, list[dict[str, Any]], str]:
        state = self.sessions.get(session_id)

        if blocked := self._safety_response(message):
            self.sessions.add_turn(session_id, message, blocked)
            return blocked, [], "safety_refusal"

        resolving_pending = state.pending_intent == "collaboration"

        if resolving_pending:
            message = f"collaboration criteria: {message}"
            state.pending_intent = None
            state.pending_question = None

        sector, location = self._extract_filters(message)

        if not resolving_pending:
            decision = self.clarification.decide(
                message, bool(sector or location)
            )
            if decision:
                state.pending_intent = decision.intent
                state.pending_question = decision.question
                self.sessions.add_turn(
                    session_id, message, decision.question
                )
                return decision.question, [], "clarification"

        # Capture history before adding the current turn.
        history = self.sessions.history(session_id)

        records, strategy = self.retrieve(message, session_id)

        if not records:
            answer = (
                "I could not find matching information in the supplied dataset."
            )
        elif not self.settings.openai_api_key:
            answer = (
                "Matching dataset records were found, but response "
                "generation is unavailable until OPENAI_API_KEY is configured."
            )
        else:
            try:
                model = ChatOpenAI(
                    model=self.settings.openai_model,
                    api_key=self.settings.openai_api_key,
                    temperature=0,
                )

                prompt = build_chat_prompt(message, records, history)
                response = (prompt | model).invoke({})
                answer = str(response.content)

            except Exception:
                logger.exception(
                    "Response generation failed for session %s",
                    session_id,
                )
                answer = (
                    "I found relevant dataset records but could not "
                    "generate a response. Please try again."
                )
                strategy = "fallback"

        # Keep the latest successfully retrieved evidence for follow-ups.
        if records:
            state.records = records

        self.sessions.add_turn(session_id, message, answer)
        return answer, records, strategy

    def history(self, session_id: str):
        return self.sessions.history(session_id)

    def clear_session(self, session_id: str) -> None:
        self.sessions.clear(session_id)

    def clarification_question(self, message: str) -> str | None:
        sector, location = self._extract_filters(message)
        decision = self.clarification.decide(
            message, bool(sector or location)
        )
        return decision.question if decision else None
