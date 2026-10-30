"""Structured LLM conversational resolution with a safe local fallback."""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from ..core.config import Settings
from ..memory import PendingClarification
from ..routing import IntentRouter
from .schemas import ConversationResolution

logger = logging.getLogger(__name__)

RESOLVER_SYSTEM_MESSAGE = """Interpret an Indian-unicorn chatbot request; do not answer it or invent facts.
Return structured output. Use previous records only if the request truly refers to them. A new explicit topic is independent.
Clarify only when an essential criterion or reference cannot be resolved. Dataset attributes are limited to the supplied records."""


class ConversationResolver:
    _REFERENCE = re.compile(r"\b(it|its|they|their|them|these|those|which of (?:these|them)|first|second|third|compare|what about|and the)\b", re.I)
    _VAGUE = re.compile(r"\b(best|good)\b.*\b(?:collaborat\w*|partner\w*)\b", re.I)

    def __init__(self, settings: Settings, router: IntentRouter) -> None:
        self.settings, self.router = settings, router

    def _model(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.settings.openai_model, api_key=self.settings.openai_api_key, temperature=0)

    def _fallback(self, message: str, previous_records: list[dict[str, Any]], pending: PendingClarification | None, *, used_fallback: bool = False) -> ConversationResolution:
        if pending:
            return ConversationResolution(resolved_query=f"{pending.original_query}. Clarification: {message}", intent=pending.intent, context_dependent=True, use_previous_results=bool(previous_records), filters=pending.filters, used_fallback=used_fallback)
        route = self.router.route(message)
        if route.filters and previous_records and self._REFERENCE.search(message):
            return ConversationResolution(resolved_query=message, intent="contextual_filter", context_dependent=True, use_previous_results=True, filters=route.filters, used_fallback=used_fallback)
        if previous_records and self._REFERENCE.search(message):
            selected = previous_records[:1] if re.search(r"\bfirst\b", message, re.I) else previous_records[1:2] if re.search(r"\bsecond\b", message, re.I) else previous_records
            name = selected[0].get("company", "") if len(selected) == 1 else ""
            return ConversationResolution(resolved_query=f"{message} for {name}" if name else message, intent="contextual", context_dependent=True, use_previous_results=True, used_fallback=used_fallback)
        if self._VAGUE.search(message):
            return ConversationResolution(status="clarify", resolved_query=message, intent="collaboration", clarification_question="What type of collaboration are you considering—for example payments, technology, logistics, or another area?", used_fallback=used_fallback)
        return ConversationResolution(resolved_query=message, intent="semantic", used_fallback=used_fallback)

    def resolve(self, message: str, history: list[Any], previous_records: list[dict[str, Any]], pending: PendingClarification | None) -> ConversationResolution:
        if not self.settings.openai_api_key:
            return self._fallback(message, previous_records, pending)
        try:
            history_payload = [{"user": str(turn.user), "assistant": str(turn.assistant)} for turn in history]
            prompt = [("system", RESOLVER_SYSTEM_MESSAGE), ("human", "History: {history}\nPrevious records: {records}\nPending clarification: {pending}\nCurrent request: {message}")]
            # json_schema is LangChain's default in this installed version, but
            # OpenAI rejects this model's dynamic filters object in that mode.
            # Tool/function calling retains Pydantic parsing without requiring
            # OpenAI's strict response_format JSON-schema subset.
            model = self._model().with_structured_output(
                ConversationResolution,
                method="function_calling",
            )
            result = model.invoke([{"role": role, "content": content.format(history=history_payload, records=previous_records[:20], pending=pending, message=message)} for role, content in prompt])
            resolution = result if isinstance(result, ConversationResolution) else ConversationResolution.model_validate(result)
            if resolution.status == "clarify" and not resolution.clarification_question:
                resolution.status = "ready"
            if not resolution.resolved_query.strip(): resolution.resolved_query = message
            return resolution
        except Exception as exc:
            logger.exception(
                "Conversation resolution failed (%s); using local fallback.",
                type(exc).__name__,
            )
            return self._fallback(
                message,
                previous_records,
                pending,
                used_fallback=True,
            )
