from pathlib import Path
import pandas as pd
import pytest

from unicorn_chatbot.data import UnicornRepository


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "unicorns.csv"
    pd.DataFrame([
        {"Company Name": "Alpha Pay", "Sector": "FinTech", "Location": "Bengaluru"},
        {"Company Name": "Beta Health", "Sector": "Health Tech", "Location": "Mumbai"},
        {"Company Name": "Alpha Pay", "Sector": "FinTech", "Location": "Bengaluru"},
    ]).to_csv(path, index=False)
    return path


@pytest.fixture
def repository(csv_path: Path) -> UnicornRepository:
    return UnicornRepository.from_csv(csv_path)

