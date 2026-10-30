"""Dataset-safe text normalization helpers."""
from __future__ import annotations

import re

import pandas as pd


def normalize_column_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")


def normalize_search_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def normalize_location_text(value: object) -> str:
    return re.sub(r"\bbangalore\b", "bengaluru", normalize_search_text(value))
