"""Thin orchestration of safety, routing, resolution, retrieval, and answers."""
from __future__ import annotations
from typing import Any

from ..conversation import ConversationResolver
from ..core.config import Settings
from ..data import UnicornRepository
from ..generation import AnswerGenerator
from ..memory import PendingClarification, SessionManager
from ..retrieval import RetrievalEngine, SemanticRetriever
from ..retrieval.company_lookup import CompanyLookup
from ..retrieval.contextual_retriever import ContextualRetriever
from ..retrieval.structured_retriever import StructuredRetriever
from ..routing import IntentRouter, RouteDecision


class ChatbotService:
    def __init__(self, repository: UnicornRepository, settings: Settings, retriever: SemanticRetriever | None = None, sessions: SessionManager | None = None, *, router: IntentRouter | None = None, resolver: ConversationResolver | None = None, generator: AnswerGenerator | None = None) -> None:
        self.repository, self.settings = repository, settings
        self.sessions = sessions or SessionManager(settings.max_session_turns)
        self.router = router or IntentRouter(repository)
        self.resolver = resolver or ConversationResolver(settings, self.router)
        self.generator = generator or AnswerGenerator(settings)
        self.engine = RetrievalEngine(CompanyLookup(repository), StructuredRetriever(repository), ContextualRetriever(repository), retriever, settings.retrieval_min_score)

    @staticmethod
    def _safety_response(message: str) -> str | None:
        text = message.casefold()
        if any(phrase in text for phrase in ("api key", "system prompt", "reveal your")):
            return "I can’t provide credentials, internal instructions, or configuration. I can help with questions supported by the dataset."
        if "use the internet" in text or "web search" in text:
            return "I can only use the supplied dataset and cannot use live web information."
        if "invent" in text and ("revenue" in text or "dataset" in text):
            return "I can’t invent missing company information. I can answer only from the supplied dataset fields."
        return None

    def retrieve(self, message: str, session_id: str = "default", use_previous_results: bool = False, original_message: str | None = None) -> tuple[list[dict[str, Any]], str]:
        state = self.sessions.get(session_id)
        decision = self.router.route(original_message or message)
        result = self.engine.retrieve(message, decision, previous_records=state.records, use_previous_results=use_previous_results)
        return result.records, result.strategy

    def answer(self, message: str, session_id: str = "default") -> tuple[str, list[dict[str, Any]], str]:
        state = self.sessions.get(session_id)
        if blocked := self._safety_response(message):
            self.sessions.add_turn(session_id, message, blocked)
            return blocked, [], "safety_refusal"
        history = self.sessions.history(session_id)
        route = self.router.route(message)
        use_context = False
        resolved_query, intent, filters = message, route.intent, route.filters
        if route.path == "conversation":
            resolution = self.resolver.resolve(message, history, state.records, state.pending_clarification)
            if resolution.status == "clarify":
                question = resolution.clarification_question or "Could you clarify what you're looking for?"
                state.pending_clarification = PendingClarification(resolution.resolved_query or message, resolution.intent, question, resolution.filters)
                self.sessions.add_turn(session_id, message, question)
                return question, [], "clarification"
            resolved_query, intent, filters = resolution.resolved_query, resolution.intent, resolution.filters
            use_context = resolution.use_previous_results
            route = RouteDecision("conversation", intent, filters=filters, reason="resolver")
        state.pending_clarification = None
        result = self.engine.retrieve(resolved_query, route, previous_records=state.records, use_previous_results=use_context)
        answer = self.generator.generate(resolved_query, result.records, history)
        if result.records: state.records = result.records
        state.previous_resolved_query, state.previous_intent, state.filters = resolved_query, intent, filters
        self.sessions.add_turn(session_id, message, answer)
        return answer, result.records, result.strategy

    def history(self, session_id: str): return self.sessions.history(session_id)
    def clear_session(self, session_id: str) -> None: self.sessions.clear(session_id)
