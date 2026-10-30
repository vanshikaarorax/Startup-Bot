from __future__ import annotations
from ..data import UnicornRepository


class CompanyLookup:
    def __init__(self, repository: UnicornRepository) -> None: self.repository = repository
    def retrieve(self, names: tuple[str, ...]) -> list[dict]:
        return [record for name in names for record in self.repository.find_companies(name)]
