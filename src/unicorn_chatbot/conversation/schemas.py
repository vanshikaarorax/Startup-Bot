"""Pydantic contract for structured conversational interpretation."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class ConversationResolution(BaseModel):
    status: Literal["ready", "clarify"] = "ready"
    resolved_query: str = ""
    intent: str = "semantic"
    context_dependent: bool = False
    use_previous_results: bool = False
    filters: dict[str, str | int] = Field(default_factory=dict)
    clarification_question: str | None = None
    used_fallback: bool = False
