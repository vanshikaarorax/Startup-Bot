from pathlib import Path

import pandas as pd

from unicorn_chatbot.config import Settings
from unicorn_chatbot.data import UnicornRepository
from unicorn_chatbot.orchestration.service import ChatbotService
from unicorn_chatbot.routing import IntentRouter


def _repository(tmp_path: Path) -> UnicornRepository:
    path = tmp_path / "unicorns.csv"
    pd.DataFrame([
        {"Company": "Alpha Pay", "Location": "Bengaluru", "Country": "India", "Stage": "Series D", "Primary Sector": "Payments", "Founded Year": 2018},
        {"Company": "Beta Health", "Location": "Mumbai", "Country": "India", "Stage": "Public", "Primary Sector": "Healthcare IT", "Founded Year": 2014},
    ]).to_csv(path, index=False)
    return UnicornRepository.from_csv(path)


def test_fast_route_skips_resolver_for_schema_backed_request(tmp_path):
    repository = _repository(tmp_path)

    class ResolverSpy:
        called = False
        def resolve(self, *_args, **_kwargs):
            self.called = True
            raise AssertionError("fast route must not call resolver")

    spy = ResolverSpy()
    service = ChatbotService(repository, Settings(), resolver=spy)
    _, sources, strategy = service.answer("Show companies in Bengaluru", "fast")
    assert strategy == "structured"
    assert sources[0]["company"] == "Alpha Pay"
    assert not spy.called


def test_router_supports_actual_country_stage_and_founded_year_fields(tmp_path):
    router = IntentRouter(_repository(tmp_path))
    assert router.route("Show Indian unicorns").filters == {"country": "India"}
    assert router.route("List Series D companies").filters == {"stage": "Series D"}
    assert router.route("Show companies founded after 2015").filters == {"founded_year": 2015, "founded_year_operator": "gt"}


def test_pending_clarification_is_explicit_and_new_company_changes_topic(tmp_path):
    service = ChatbotService(_repository(tmp_path), Settings())
    _, _, strategy = service.answer("Which company is best to collaborate with?", "pending")
    assert strategy == "clarification"
    assert service.sessions.get("pending").pending_clarification is not None
    service.answer("payments", "pending")
    assert service.sessions.get("pending").pending_clarification is None
    _, sources, strategy = service.answer("Tell me about Beta Health", "pending")
    assert strategy == "company_lookup"
    assert [source["company"] for source in sources] == ["Beta Health"]
