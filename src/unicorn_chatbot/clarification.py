"""Deterministic clarification decisions for underspecified collaboration requests."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClarificationDecision:
    intent: str
    question: str


class ClarificationPolicy:
    def decide(self, message: str, has_constraints: bool) -> ClarificationDecision | None:
        lowered = message.casefold()
        if ("collaborat" in lowered or "partner" in lowered) and not has_constraints:
            return ClarificationDecision(
                "collaboration",
                "What type of collaboration are you considering—for example payments, technology, logistics, or another area?",
            )
        return None
