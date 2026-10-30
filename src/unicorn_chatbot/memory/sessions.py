"""Bounded, in-memory, session-isolated conversation state."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass(slots=True)
class Turn:
    user: str
    assistant: str


@dataclass(slots=True)
class PendingClarification:
    original_query: str
    intent: str
    question: str
    filters: dict[str, str | int] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationState:
    turns: list[Turn] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    previous_resolved_query: str | None = None
    previous_intent: str | None = None
    filters: dict[str, str | int] = field(default_factory=dict)
    pending_clarification: PendingClarification | None = None


class SessionManager:
    def __init__(self, max_turns: int) -> None:
        self.max_turns, self._sessions, self._lock = max_turns, {}, RLock()

    def get(self, session_id: str) -> ConversationState:
        with self._lock: return self._sessions.setdefault(session_id, ConversationState())

    def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            state = self.get(session_id)
            state.turns.append(Turn(user, assistant))
            del state.turns[:-self.max_turns]

    def history(self, session_id: str) -> list[Turn]:
        with self._lock: return list(self.get(session_id).turns)

    def clear(self, session_id: str) -> None:
        with self._lock: self._sessions.pop(session_id, None)
