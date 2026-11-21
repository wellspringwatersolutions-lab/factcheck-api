import time

from factcheck.cleanup import focus_query, prepare_answer
from factcheck.extract import heuristic_extract

from tests.conftest import client, make_service

GARLIC = """
Garlic extract lowered systolic blood pressure by 8 mmHg in adults with hypertension.
| Supplement | Dose | Average Reduction | Source |
| Garlic | 600 mg | 5 mmHg | PMC - NIH |
Source: iHerb Clinical meta-analyses of garlic show a reduction in blood pressure. PMC - NIH Average Reduction: 4 mmHg.
GoodRx lists a typical dose beside the trial.
Want to compare garlic brands? Yes
"""


def test_junk_and_table_rows_are_stripped_before_search():
    prepared = prepare_answer(GARLIC)
    assert "iHerb" not in prepared
    assert "PMC" not in prepared
    assert "GoodRx" not in prepared
    assert "Want to compare" not in prepared
    assert "600 mg" in prepared
    claims = heuristic_extract(GARLIC, None)
    blob = " ".join(claim.text for claim in claims)
    assert "iHerb" not in blob
    assert "PMC" not in blob
    assert "Want to compare" not in blob
    assert any("600 mg" in claim.text and "Garlic" in claim.text for claim in claims)
    query = focus_query(blob)
    assert "Source:" not in query
    assert len(query) <= 160
    assert "iHerb" not in query


class _SlowRetriever:
    def __init__(self, delay: float):
        self.delay = delay
        self.queries: list[str] = []
        self.cost_usd = 0.0
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 6, scientific: bool = False, categories=None, **_ignored):
        time.sleep(self.delay)
        self.queries.append(query)
        return []


_TOPICS = [
    "The Eiffel Tower is located in Paris.",
    "Canberra is the capital city of Australia.",
    "Mount Everest rises 8849 meters above sea level.",
    "The Nile is a major river in northeastern Africa.",
    "Tokyo is the capital city of Japan.",
    "Gold is a dense solid metal at room temperature.",
    "Shakespeare wrote the tragedy Hamlet.",
    "Oxygen is a chemical element used in respiration.",
    "The Amazon River flows through South America.",
    "Aspirin reduces the risk of a heart attack in adults.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Ada Lovelace wrote notes on the Analytical Engine.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Moon is a natural satellite of Earth.",
    "Penicillin was discovered by Alexander Fleming.",
]


def _facts(count: int) -> list[str]:
    return _TOPICS[:count]


def test_claims_are_capped_and_the_rest_are_reported():
    service, retriever, _, _ = make_service(retriever=_SlowRetriever(0))
    service.settings.max_checked_claims = 4
    result = service.verify(text=None, claims=_facts(8), client=client(), charge=False)
    assert len(result.claims) == 8
    skipped = [claim for claim in result.claims if "highest-stakes" in claim.reasoning]
    assert len(skipped) == 4
    assert all(claim.verdict == "Unverified" for claim in skipped)
    assert len(retriever.queries) == 4


def test_clusters_run_concurrently():
    service, retriever, _, _ = make_service(retriever=_SlowRetriever(0.25))
    service.settings.max_checked_claims = 4
    service.settings.check_concurrency = 4
    started = time.perf_counter()
    service.verify(text=None, claims=_facts(4), client=client(fingerprint="speed"), charge=False)
    elapsed = time.perf_counter() - started
    assert len(retriever.queries) == 4
    assert elapsed < 0.7


def test_time_budget_returns_unverified_instead_of_blocking():
    service, retriever, _, _ = make_service(retriever=_SlowRetriever(1.0))
    service.settings.request_budget_seconds = 0
    service.settings.max_checked_claims = 4
    started = time.perf_counter()
    result = service.verify(text=None, claims=_facts(4), client=client(fingerprint="budget"), charge=False)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5
    assert retriever.queries == []
    assert all("time budget" in claim.reasoning for claim in result.claims)


def test_fifteen_claims_stay_fast_when_retrieval_is_slow():
    service, retriever, _, _ = make_service(retriever=_SlowRetriever(0.2))
    service.settings.max_checked_claims = 6
    service.settings.check_concurrency = 4
    started = time.perf_counter()
    result = service.verify(text=None, claims=_facts(15), client=client(fingerprint="fifteen"), charge=False)
    elapsed = time.perf_counter() - started
    assert len(result.claims) == 15
    assert len(retriever.queries) == 6
    assert elapsed < 1.2
    assert sum("Not checked" in claim.reasoning for claim in result.claims) == 9


def test_disconnected_client_does_not_consume_the_free_check():
    service, retriever, _, _ = make_service()
    identity = client(fingerprint="drop")
    service.verify(text=None, claims=["2 + 2 = 4"], client=identity, disconnect_check=lambda: True)
    assert retriever.queries == []
    again = service.verify(text=None, claims=["2 + 2 = 4"], client=identity)
    assert again.claims[0].verdict == "Supported"
