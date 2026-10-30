"""Read-only repository for the supplied unicorn CSV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .normalizers import normalize_column_name, normalize_location_text, normalize_search_text


class DatasetError(ValueError):
    """Raised when the configured dataset cannot safely be used."""


@dataclass(frozen=True, slots=True)
class DatasetReport:
    rows: int
    columns: int
    missing_values: int
    duplicate_records: int


class UnicornRepository:
    """Deterministic access that preserves source values in returned records."""

    FILTER_COLUMNS = {
        "sector": ("primary_sector", "sector", "industry", "category", "vertical"),
        "location": ("location", "headquarters", "city", "headquarter"),
        "country": ("country",),
        "stage": ("stage", "funding_stage"),
        "founded_year": ("founded_year",),
    }

    def __init__(self, frame: pd.DataFrame, report: DatasetReport) -> None:
        self.frame = frame
        self.report = report
        self._name_column = self._find_name_column()

    @classmethod
    def from_csv(cls, path: Path) -> "UnicornRepository":
        if not path.is_file():
            raise DatasetError(f"Dataset not found at '{path}'. Download the supplied Kaggle CSV manually and set DATASET_PATH (default: data/unicorns.csv).")
        try:
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                frame = pd.read_csv(path, encoding="cp1252")
        except Exception as exc:
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
        report = DatasetReport(len(cleaned), len(cleaned.columns), int(cleaned.isna().sum().sum() + (cleaned == "").sum().sum()), duplicates)
        repository = cls(cleaned, report)
        if repository._name_column is None:
            raise DatasetError("Dataset must contain a company-name column (for example 'Company' or 'Company Name').")
        return repository

    def _find_name_column(self) -> str | None:
        return next((name for name in ("company", "company_name", "name", "startup", "startup_name") if name in self.frame.columns), None)

    @property
    def name_column(self) -> str:
        if self._name_column is None:
            raise DatasetError("No company name column is available.")
        return self._name_column

    def available_columns(self) -> list[str]:
        return list(self.frame.columns)

    def _column_for(self, kind: str) -> str | None:
        return next((column for column in self.FILTER_COLUMNS[kind] if column in self.frame.columns), None)

    def values_for(self, kind: str) -> list[str]:
        column = self._column_for(kind)
        return [] if column is None else self.frame[column].dropna().astype(str).unique().tolist()

    @staticmethod
    def _json_value(value: object) -> Any:
        return None if pd.isna(value) else value

    def _records(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [{key: self._json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]

    def find_companies(self, name: str) -> list[dict[str, Any]]:
        query = normalize_search_text(name)
        if not query:
            return []
        return self._records(self.frame[self.frame[self.name_column].map(normalize_search_text) == query])

    def mentioned_companies(self, message: str) -> list[str]:
        text = normalize_search_text(message)
        return [name for name in self.values_for_name() if normalize_search_text(name) in text]

    def values_for_name(self) -> list[str]:
        return self.frame[self.name_column].dropna().astype(str).unique().tolist()

    def filter(self, *, sector: str | None = None, location: str | None = None, country: str | None = None, stage: str | None = None, founded_year: int | None = None, founded_year_operator: str | None = None) -> list[dict[str, Any]]:
        filtered = self.frame
        for kind, term in (("sector", sector), ("location", location), ("country", country), ("stage", stage)):
            if not term or not term.strip():
                continue
            column = self._column_for(kind)
            if column is None:
                return []
            normalizer = normalize_location_text if kind == "location" else normalize_search_text
            wanted = normalizer(term)
            filtered = filtered[filtered[column].map(normalizer).map(lambda value: wanted in value)]
        if founded_year is not None:
            column = self._column_for("founded_year")
            if column is None:
                return []
            years = pd.to_numeric(filtered[column], errors="coerce")
            match founded_year_operator or "eq":
                case "gt": filtered = filtered[years > founded_year]
                case "gte": filtered = filtered[years >= founded_year]
                case "lt": filtered = filtered[years < founded_year]
                case "lte": filtered = filtered[years <= founded_year]
                case _: filtered = filtered[years == founded_year]
        return self._records(filtered)

    def all_records(self) -> list[dict[str, Any]]:
        return self._records(self.frame)
