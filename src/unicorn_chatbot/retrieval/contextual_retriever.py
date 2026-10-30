"""Selection and filtering of already retrieved records."""
from __future__ import annotations
import re
from ..data import UnicornRepository, normalize_location_text, normalize_search_text


class ContextualRetriever:
    def __init__(self, repository: UnicornRepository) -> None: self.repository = repository
    def retrieve(self, message: str, records: list[dict], filters: dict[str, str | int]) -> list[dict]:
        text, selected = message.casefold(), records
        if re.search(r"\bfirst\s+two\b", text): selected = records[:2]
        elif re.search(r"\bsecond\b", text): selected = records[1:2]
        elif re.search(r"\bfirst\b", text): selected = records[:1]
        for kind in ("sector", "location", "country", "stage"):
            value = filters.get(kind)
            column = self.repository._column_for(kind)
            if value and column:
                normalizer = normalize_location_text if kind == "location" else normalize_search_text
                target = normalizer(value)
                selected = [record for record in selected if target in normalizer(record.get(column, ""))]
        return selected
