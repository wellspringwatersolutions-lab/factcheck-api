import httpx

from factcheck.retrievers import DesearchRetriever
from factcheck.types import Evidence

from tests.conftest import client, make_service


def test_the_fallback_query_names_the_four_fact_checkers_and_counts_as_one_call():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params.get("query") or "")
        return httpx.Response(200, json={"data": []})

    desearch = DesearchRetriever("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    desearch.search_fact_checkers("The city council raised taxes in 2019.")
    assert len(calls) == 1
    query = calls[0]
    assert "site:snopes.com" in query
    assert "site:politifact.com" in query
    assert "site:factcheck.org" in query
    assert "site:apnews.com" in query
    assert "ap-fact-check" in query


def test_an_unverified_claim_asks_once_and_a_google_hit_does_not():
    class Web:
        def __init__(self):
            self.fact_checks = []
            self.queries = []

        def search(self, query, **_ignored):
            self.queries.append(query)
            return []

        def search_fact_checkers(self, claim):
            self.fact_checks.append(claim)
            return []

    service, retriever, _, _ = make_service(retriever=Web())
    service.verify(text=None, claims=["The city council raised property taxes in 2019."], client=client(), charge=False)
    assert len(retriever.fact_checks) == 1

    class Google:
        def search(self, query, *, limit=4):
            return [
                Evidence(
                    url="https://www.snopes.com/fact-check/taxes",
                    title="Snopes",
                    snippet="Snopes rating: True. The city council raised property taxes in 2019.",
                    page_text="Snopes rating: True. The city council raised property taxes in 2019.",
                    provider="google_factcheck",
                    source_type="fact_checker",
                    grade="B",
                )
            ]

    covered, covered_web, _, _ = make_service(retriever=Web())
    covered.google_factcheck = Google()
    covered.verify(text=None, claims=["The city council raised property taxes in 2019."], client=client(), charge=False)
    assert covered_web.fact_checks == []
