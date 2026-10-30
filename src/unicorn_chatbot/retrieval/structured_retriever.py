from __future__ import annotations
from ..data import UnicornRepository


class StructuredRetriever:
    def __init__(self, repository: UnicornRepository) -> None: self.repository = repository
    def retrieve(self, filters: dict[str, str | int]) -> list[dict]:
        return self.repository.filter(**filters)  # type: ignore[arg-type]
