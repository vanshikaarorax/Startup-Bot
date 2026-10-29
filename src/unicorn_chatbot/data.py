"""CSV loading, validation, normalization, and deterministic record access."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd


class DatasetError(ValueError):
    """Raised when the configured dataset cannot safely be used."""


def normalize_column_name(value: str) -> str:
    """Convert a CSV heading to a predictable snake_case name."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")


def normalize_search_text(value: object) -> str:
    """Normalize case and whitespace for matching without altering source values."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def normalize_location_text(value: object) -> str:
    """Normalize common city spelling variants for location matching only."""
    text = normalize_search_text(value)
    return re.sub(r"\bbangalore\b", "bengaluru", text)


@dataclass(frozen=True, slots=True)
class DatasetReport:
    rows: int
    columns: int
    missing_values: int
    duplicate_records: int


class UnicornRepository:
    """Read-only, deterministic access to normalized dataset records.

    Exact duplicate rows are removed, retaining the first occurrence in CSV order.
    Original cell values are preserved in the returned record dictionaries.
    """

    def __init__(self, frame: pd.DataFrame, report: DatasetReport) -> None:
        self.frame = frame
        self.report = report
        self._name_column = self._find_name_column()

    @classmethod
    def from_csv(cls, path: Path) -> "UnicornRepository":
        if not path.is_file():
            raise DatasetError(
                f"Dataset not found at '{path}'. Download the supplied Kaggle CSV manually "
                "and set DATASET_PATH (default: data/unicorns.csv)."
            )
        try:
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                frame = pd.read_csv(path, encoding="cp1252")
        except Exception as exc:  # pandas raises several parser/IO exception types
            raise DatasetError(f"Could not parse dataset '{path}': {exc}") from exc
        if frame.empty or not len(frame.columns):
            raise DatasetError(f"Dataset '{path}' is empty or has no columns.")

        columns = [normalize_column_name(str(column)) for column in frame.columns]
        if len(set(columns)) != len(columns):
            raise DatasetError("Column normalization produced duplicate column names.")
        frame.columns = columns
        frame = frame.map(lambda value: value.strip() if isinstance(value, str) else value)
        duplicates = int(frame.duplicated().sum())
        cleaned = frame.drop_duplicates().reset_index(drop=True)
        missing = int(cleaned.isna().sum().sum() + (cleaned == "").sum().sum())
        report = DatasetReport(len(cleaned), len(cleaned.columns), missing, duplicates)
        repository = cls(cleaned, report)
        if repository._name_column is None:
            raise DatasetError(
                "Dataset must contain a company-name column (for example 'Company' or 'Company Name')."
            )
        return repository

    def _find_name_column(self) -> str | None:
        choices = ("company", "company_name", "name", "startup", "startup_name")
        return next((name for name in choices if name in self.frame.columns), None)

    @property
    def name_column(self) -> str:
        if self._name_column is None:
            raise DatasetError("No company name column is available.")
        return self._name_column

    def available_columns(self) -> list[str]:
        return list(self.frame.columns)

    def _column_for(self, kind: str) -> str | None:
        aliases = {
            "sector": ("sector", "industry", "category", "vertical"),
            "location": ("location", "headquarters", "city", "headquarter"),
        }
        return next((column for column in aliases[kind] if column in self.frame.columns), None)

    def _records(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [{key: self._json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]

    @staticmethod
    def _json_value(value: object) -> Any:
        return None if pd.isna(value) else value

    def find_companies(self, name: str) -> list[dict[str, Any]]:
        query = normalize_search_text(name)
        if not query:
            return []
        matches = self.frame[self.frame[self.name_column].map(normalize_search_text) == query]
        return self._records(matches)

    def filter(self, sector: str | None = None, location: str | None = None) -> list[dict[str, Any]]:
        """Return records whose sector/location contains the normalized requested term."""
        filtered = self.frame
        for kind, term in (("sector", sector), ("location", location)):
            if not term or not term.strip():
                continue
            column = self._column_for(kind)
            if column is None:
                return []
            normalizer = normalize_location_text if kind == "location" else normalize_search_text
            wanted = normalizer(term)
            filtered = filtered[filtered[column].map(normalizer).map(lambda value: wanted in value)]
        return self._records(filtered)

    def all_records(self) -> list[dict[str, Any]]:
        return self._records(self.frame)

