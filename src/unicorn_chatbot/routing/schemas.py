"""Deterministic route decision contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class RouteDecision:
    path: Literal["fast", "conversation"]
    intent: str
    filters: dict[str, str | int] = field(default_factory=dict)
    company_names: tuple[str, ...] = ()
    reason: str = ""
