from pathlib import Path
import pytest

from unicorn_chatbot.data import DatasetError, UnicornRepository, normalize_search_text


def test_load_normalizes_columns_and_deduplicates(repository):
    assert repository.available_columns() == ["company_name", "sector", "location"]
    assert repository.report.rows == 2
    assert repository.report.duplicate_records == 1


def test_missing_dataset_is_actionable(tmp_path: Path):
    with pytest.raises(DatasetError, match="Download"):
        UnicornRepository.from_csv(tmp_path / "missing.csv")


def test_normalization_and_company_lookup(repository):
    assert normalize_search_text("  FIN Tech ") == "fin tech"
    assert repository.find_companies(" alpha pay ")[0]["location"] == "Bengaluru"


def test_sector_location_and_combined_filters(repository):
    assert len(repository.filter(sector="fintech")) == 1
    assert len(repository.filter(location="mumbai")) == 1
    assert repository.filter(sector="fintech", location="Mumbai") == []
    assert repository.filter(location="Bangalore")[0]["company_name"] == "Alpha Pay"


def test_empty_search_result(repository):
    assert repository.find_companies("unknown") == []

