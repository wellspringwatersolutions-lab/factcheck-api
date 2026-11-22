import time

import httpx
from fastapi.testclient import TestClient

from factcheck.app import create_app
from factcheck.claim_cache import ClaimCache
from factcheck.early import factcheck_short_circuit, search_is_enough
from factcheck.extract import LiveExtractor
from factcheck.llm import ScriptedLLM
from factcheck.retrievers import DesearchRetriever
from factcheck.schemas import ClaimOut
from factcheck.types import Evidence, UsageMeter

from tests.conftest import client, make_service, settings


def test_desearch_queries_run_together():
    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.08)
        return httpx.Response(200, json={"data": [{"link": "https://example.com/a", "title": "t", "snippet": "s"}]})

    desearch = DesearchRetriever("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    queries = [f"query number {i} about a distinct claim" for i in range(8)]
    started = time.monotonic()
    desearch.search_web_and_news(queries)
    elapsed = time.monotonic() - started
    # The same 8 calls, run one after another, took 0.646s before this change.
    assert elapsed < 0.35, elapsed


def test_one_fact_checker_rating_stops_the_rest_of_the_queries():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params.get("query"))
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "link": "https://www.snopes.com/fact-check/tower",
                        "title": "Snopes",
                        "snippet": "Snopes rating: False. The Eiffel Tower is not in Berlin.",
                    }
                ]
            },
        )

    desearch = DesearchRetriever("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    hits = desearch.search_web_and_news(
        ["eiffel tower berlin", "eiffel tower location"],
        stop_when=lambda found: search_is_enough("The Eiffel Tower is in Berlin.", found),
    )
    assert calls == ["eiffel tower berlin"]
    assert factcheck_short_circuit("The Eiffel Tower is in Berlin.", hits)


def test_two_reputable_sources_that_agree_stop_the_search():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params.get("query"))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"link": "https://www.reuters.com/tower", "title": "Reuters", "snippet": "The Eiffel Tower is located in Paris."},
                    {"link": "https://www.bbc.com/tower", "title": "BBC", "snippet": "The Eiffel Tower is located in Paris."},
                ]
            },
        )

    desearch = DesearchRetriever("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    desearch.search_web_and_news(
        ["first query", "second query"],
        stop_when=lambda found: search_is_enough("The Eiffel Tower is located in Paris.", found),
    )
    assert calls == ["first query"]


def test_google_fact_check_runs_first_and_can_skip_desearch():
    class Google:
        def __init__(self):
            self.queries = []

        def search(self, query, *, limit=6):
            self.queries.append(query)
            return [
                Evidence(
                    url="https://www.snopes.com/fact-check/tower",
                    title="Snopes",
                    snippet="Snopes rating: False. The Eiffel Tower is not in Berlin.",
                    page_text="Snopes rating: False. The Eiffel Tower is not in Berlin.",
                    provider="google_factcheck",
                    source_type="fact_checker",
                    grade="B",
                )
            ]

    service, retriever, _, _ = make_service()
    google = Google()
    service.google_factcheck = google
    result = service.verify(
        text=None,
        claims=["The Eiffel Tower is in Berlin."],
        client=client(),
        charge=False,
    )
    assert google.queries
    assert retriever.queries == []
    assert result.claims[0].verdict in {"Contradicted", "Conflicting", "Unverified", "likely_true", "Supported"}


def test_a_missing_google_key_leaves_fact_check_disabled():
    service, retriever, _, _ = make_service()
    assert service.google_factcheck is None
    service.verify(text=None, claims=["The Eiffel Tower is located in Paris."], client=client(), charge=False)
    assert retriever.queries


def test_a_cached_claim_is_reused_and_marked_in_usage(tmp_path):
    service, retriever, _, _ = make_service()
    service.claim_cache = ClaimCache(path=str(tmp_path / "claims.json"))
    text = "The Eiffel Tower is located in Paris."
    first = service.verify(text=None, claims=[text], client=client(), charge=False)
    queries = list(retriever.queries)
    second = service.verify(text=None, claims=[text], client=client(fingerprint="again"), charge=False)
    assert second.usage.cached_claims == 1
    assert retriever.queries == queries
    assert second.claims[0].verdict == first.claims[0].verdict
    assert second.claims[0].sources == first.claims[0].sources
    reloaded = ClaimCache(path=str(tmp_path / "claims.json"))
    assert reloaded.get(text).verdict == first.claims[0].verdict
    service.claim_cache.put(
        ClaimOut(text="This expires immediately.", claim_type="fact", claim_category="T5", verdict="Unverified", reasoning="Checked."),
        ttl=-1,
    )
    assert service.claim_cache.get("This expires immediately.") is None


def test_job_claims_stream_as_they_finish():
    service, _, _, cfg = make_service()
    app = create_app(settings=cfg, service=service)
    with TestClient(app) as http:
        created = http.post(
            "/check",
            json={"text": "The Pentagon gave commanders 90 days to harden bases.", "async": True},
            headers={"X-Client-Fingerprint": "stream-a"},
        )
        job_id = created.json()["job_id"]
        for _ in range(50):
            if http.get(f"/v1/jobs/{job_id}").json()["status"] == "done":
                break
            time.sleep(0.05)
        with http.stream("GET", f"/v1/jobs/{job_id}/events") as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
        assert "event: claim" in body
        assert "event: done" in body
        assert "Pentagon" in body or "commanders" in body
        missing = http.get("/v1/jobs/missing-job/events")
        assert missing.status_code == 404


def test_progress_includes_each_finished_claim():
    service, _, _, _ = make_service()
    seen = []

    def progress(done, total, claims=None):
        seen.append((done, total, [claim.text for claim in (claims or [])]))

    service.verify(
        text=None,
        claims=["The Eiffel Tower is located in Paris.", "Canberra is the capital city of Australia."],
        client=client(),
        charge=False,
        on_progress=progress,
    )
    assert seen
    assert seen[-1][0] == seen[-1][1] == 2
    assert any(len(texts) >= 1 for _, _, texts in seen)


def test_fast_model_is_optional_and_affine_is_not_selected():
    extractor = LiveExtractor(ScriptedLLM(["{}"]), "deepseek-ai/DeepSeek-V3.2-TEE", UsageMeter(), settings())
    assert extractor.extraction_models() == [
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "deepseek-ai/DeepSeek-V3.2-TEE",
    ]
    assert all("affine" not in name.lower() for name in extractor.extraction_models())
    fast = LiveExtractor(
        ScriptedLLM(["{}"]),
        "deepseek-ai/DeepSeek-V3.2-TEE",
        UsageMeter(),
        settings(fast_model="Qwen/Qwen3.5-27B-TEE"),
    )
    assert fast.extraction_models()[0] == "Qwen/Qwen3.5-27B-TEE"
    assert "DeepSeek-V3.2" in fast.extraction_models()[-1]
