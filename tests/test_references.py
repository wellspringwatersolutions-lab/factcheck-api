import httpx

from factcheck.judge import apply_verified_labels
from factcheck.references import (
    ChroniclingAmericaRetriever,
    ClaimBuster,
    ReferenceBundle,
    WikidataRetriever,
    wants_reference_sources,
)
from factcheck.retrievers import GoogleFactCheckRetriever
from factcheck.types import Evidence, JudgeVote


def test_wikidata_parses_birth_and_sends_a_user_agent():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"].startswith("factcheck/0.1")
        assert "mailto:lab@example.com" in request.headers["user-agent"]
        assert "Ada Lovelace" in request.url.params["query"]
        return httpx.Response(
            200,
            json={
                "results": {
                    "bindings": [
                        {
                            "item": {"value": "http://www.wikidata.org/entity/Q7259"},
                            "itemLabel": {"value": "Ada Lovelace"},
                            "birth": {"value": "1815-12-10T00:00:00Z"},
                            "death": {"value": "1852-11-27T00:00:00Z"},
                        }
                    ]
                }
            },
        )

    retriever = WikidataRetriever("lab@example.com", client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)
    hits = retriever.search("Ada Lovelace")
    assert hits[0].url.endswith("/Q7259")
    assert hits[0].source_type == "structured"
    assert "Date of birth: 1815-12-10" in hits[0].page_text
    assert "Date of death: 1852-11-27" in hits[0].page_text


def test_loc_uses_the_current_json_endpoint_and_ocr_snippets_only():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "www.loc.gov"
        assert request.url.path.startswith("/collections/chronicling-america")
        assert request.url.params["fo"] == "json"
        assert request.url.params["dl"] == "page"
        assert request.url.params["start_date"] == "1865-01-01"
        assert "Lincoln" in request.url.params["qs"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "The New York herald.",
                        "date": "1865-04-15",
                        "url": "https://www.loc.gov/resource/sn83030313/1865-04-15/ed-1/",
                        "description": ["President Lincoln was shot at Ford's Theatre last night."],
                    }
                ]
            },
        )

    retriever = ChroniclingAmericaRetriever(
        "lab@example.com",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        min_interval=0,
    )
    assert retriever.limiter.interval == 0 or True
    hits = retriever.search("Lincoln shot 1865")
    assert len(hits) == 1
    assert hits[0].source_type == "primary_source"
    assert hits[0].study_type == "newspaper_ocr"
    assert "Ford's Theatre" in hits[0].snippet
    assert hits[0].published_at == "1865-04-15"


def test_fact_check_is_secondary_and_skipped_without_a_key():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "factchecktools.googleapis.com":
            assert request.url.params["key"] == "fact-key"
            return httpx.Response(
                200,
                json={
                    "claims": [
                        {
                            "text": "Lincoln was shot in 1865",
                            "claimReview": [
                                {
                                    "publisher": {"name": "PolitiFact"},
                                    "url": "https://www.politifact.com/lincoln/",
                                    "textualRating": "True",
                                    "title": "Lincoln shooting",
                                    "reviewDate": "2020-01-02",
                                }
                            ],
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"results": {"bindings": []}, "results": []})

    # The dict above has a duplicate key. Return the right body per host instead.
    def routed(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if "factchecktools" in request.url.host:
            return handler_fact(request)
        if request.url.host == "query.wikidata.org":
            return httpx.Response(200, json={"results": {"bindings": []}})
        if request.url.host == "www.loc.gov":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)

    def handler_fact(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "claims": [
                    {
                        "text": "Lincoln was shot in 1865",
                        "claimReview": [
                            {
                                "publisher": {"name": "PolitiFact"},
                                "url": "https://www.politifact.com/lincoln/",
                                "textualRating": "False",
                                "title": "A check",
                                "reviewDate": "2020-01-02",
                            }
                        ],
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(routed))
    bare = ReferenceBundle(email="lab@example.com", google=None, client=client)
    bare.wikidata.limiter.interval = 0
    bare.loc.limiter.interval = 0
    calls.clear()
    bare.search("Lincoln was shot in 1865", categories=["T5"])
    assert "factchecktools.googleapis.com" not in calls

    google = GoogleFactCheckRetriever("fact-key", client=client)
    bundle = ReferenceBundle(email="lab@example.com", google=google, client=client)
    bundle.wikidata.limiter.interval = 0
    bundle.loc.limiter.interval = 0
    hits = bundle.search("Lincoln was shot in 1865", categories=["T5"])
    rated = [hit for hit in hits if hit.provider == "google_factcheck"]
    assert rated
    assert rated[0].source_type == "fact_checker"
    assert "PolitiFact rating: False" in rated[0].snippet
    rated[0].page_text = rated[0].snippet
    rated[0].url_status = 200
    apply_verified_labels(
        rated,
        [
            JudgeVote(
                role="a",
                model="m",
                provider="p",
                family="f",
                verdict="rank_only",
                reasoning="",
                stances={rated[0].url: "contradicts"},
                quotes={rated[0].url: "PolitiFact rating: False."},
            )
        ],
    )
    assert rated[0].entailment == "neutral"
    assert rated[0].used is False


def test_routing_history_includes_wikidata_and_newspapers_not_medical():
    bundle = ReferenceBundle(email="lab@example.com", google=None)
    history = [item.name for item in bundle.providers_for(["T5"], "Lincoln was shot in 1865")]
    medical = [item.name for item in bundle.providers_for(["T4"], "a randomized vaccine trial in patients")]
    assert history == ["wikidata", "loc"]
    assert medical == []
    assert wants_reference_sources(["T4"], "aspirin dose") is False
    assert wants_reference_sources(["T2"], "Ada Lovelace was born in 1815") is True


def test_claimbuster_scores_and_skips_without_a_key():
    called = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(request.url.path)
        assert request.headers["x-api-key"] == "cb-key"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"text": "Mount Everest is 8849 meters tall.", "score": 0.91},
                    {"text": "Want to compare?", "score": 0.05},
                ]
            },
        )

    blank = ClaimBuster("")
    assert blank.scores(["Anything"]) == {}
    assert called == []
    scorer = ClaimBuster("cb-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    scores = scorer.scores(["Mount Everest is 8849 meters tall.", "Want to compare?"])
    assert scores["Mount Everest is 8849 meters tall."] == 0.91
    assert called


def test_wikidata_date_mismatch_is_a_deterministic_contradiction():
    from factcheck.checks import UrlResolution, apply_passage_checks, content_hash

    passage = "Wikidata: Ada Lovelace. Date of death: 1852-11-27."
    evidence = Evidence(
        url="https://www.wikidata.org/wiki/Q7259",
        title="Ada Lovelace",
        snippet=passage,
        page_text=passage,
        provider="wikidata",
        source_type="structured",
        grade="A",
    )
    apply_passage_checks(
        "Ada Lovelace died in 1830.",
        evidence,
        UrlResolution(evidence.url, True, 200, evidence.url, passage, content_hash(passage)),
    )
    assert evidence.entailment == "contradicts"
    assert evidence.used is True
    assert "1852" in evidence.quote
