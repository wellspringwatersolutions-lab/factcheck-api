import httpx

from factcheck.aggregate import decide
from factcheck.credibility import grade_for
from factcheck.extract import select_key_points
from factcheck.extract import heuristic_extract
from factcheck.queries import heuristic_queries, news_window
from factcheck.retrievers import DesearchRetriever
from factcheck.review import TwoPassReview, run_review
from factcheck.types import Evidence
from factcheck.web_report import to_check_report

from tests.conftest import client, make_service
from tests.test_review import _result, _reviewer


def test_news_window_follows_the_claim_and_queries_are_keywords():
    assert news_window("In October 2026 France saw a supermarket refrigeration cyberattack.") == {
        "start_date": "2026-10-01",
        "end_date": "2026-10-31",
    }
    assert news_window("The Pentagon gave commanders 90 days to harden bases.") is None
    queries = heuristic_queries("In October 2026 France reported a European supermarket refrigeration cyberattack.")
    assert queries
    blob = " ".join(queries)
    assert "In " not in blob
    assert "October" in blob and "2026" in blob
    assert "France" in blob
    assert blob != "In October France European 2026 10"
    assert all(len(query.split()) <= 10 for query in queries)


def test_desearch_runs_web_and_news_without_a_week_window():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params)))
        return httpx.Response(
            200,
            json={"data": [{"link": f"https://example.com/{request.url.path.strip('/')}", "title": "Story", "snippet": "French supermarkets lost refrigeration."}]},
        )

    retriever = DesearchRetriever("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    hits = retriever.search_web_and_news(
        ["France supermarket refrigeration October 2026", "France cyberattack refrigeration"],
        window=news_window("In October 2026 France saw a supermarket refrigeration cyberattack."),
    )
    paths = [path for path, _params in seen]
    assert paths.count("/web") == 2
    assert paths.count("/news") == 2
    news = [params for path, params in seen if path == "/news"]
    assert all(params["start_date"] == "2026-10-01" and params["end_date"] == "2026-10-31" for params in news)
    assert all("time" not in params for _path, params in seen)
    assert len(hits) == 2
    assert retriever.retrieval_calls == 4


def test_one_reputable_source_is_likely_true_and_two_still_support():
    quote = "French supermarket refrigerators were knocked offline for two days after the intrusion in October 2026."
    one = Evidence(
        url="https://www.reuters.com/france-hack",
        title="France hack",
        snippet=quote,
        page_text=quote,
        quote=quote,
        entailment="entails",
        used=True,
        grade=grade_for("https://www.reuters.com/france-hack"),
        source_type="secondary",
        origin_id="reuters.com",
    )
    decision = decide([one], category="T6", claim_text="French supermarket refrigerators were knocked offline after a cyberattack.")
    assert decision.verdict == "likely_true"
    second = Evidence(
        url="https://www.bbc.com/france-hack",
        title="France hack",
        snippet=quote,
        page_text=quote,
        quote=quote,
        entailment="entails",
        used=True,
        grade="B",
        source_type="secondary",
        origin_id="bbc.com",
    )
    both = decide([one, second], category="T6", claim_text="A French supermarket refrigeration system was hacked in 2026.")
    assert both.verdict == "Supported"
    weak = Evidence(
        url="https://random-shop.example/blog",
        title="Blog",
        snippet=quote,
        page_text=quote,
        quote=quote,
        entailment="entails",
        used=True,
        grade="E",
        source_type="secondary",
        origin_id="random-shop.example",
    )
    assert decide([weak], category="T6", claim_text="French supermarket refrigerators were knocked offline.").verdict == "Unverified"


def test_key_points_are_capped_and_a_long_paste_still_retrieves_each_one():
    sentences = [
        f"The council raised property taxes by {index} percent in 2019 after a long public debate."
        for index in range(1, 9)
    ]
    extracted = heuristic_extract(" ".join(sentences), None)
    assert len(extracted) >= 7
    assert len(select_key_points(extracted, 6)) == 6
    service, retriever, _, _ = make_service()
    result = service.verify(text=" ".join(sentences), claims=None, client=client(), charge=False, budget=120)
    assert 1 <= len(result.claims) <= 6
    assert retriever.queries
    assert all("time budget" not in claim.reasoning for claim in result.claims)
    assert result.usage.retrieval_calls >= 1
    report = to_check_report(result, " ".join(sentences))
    assert report["usage"]["retrieval_calls"] >= 1
    assert "passes" in report["review"]
    assert "estimated_cost_usd" in report["usage"]


def test_a_failed_review_pass_still_fills_the_review_field():
    reviewer, meter = _reviewer(["not json", '{"corrections": []}'])
    result = _result()
    run_review(result, reviewer=reviewer, deadline=None, meter=meter)
    assert result.review.passes
    assert result.review.signed_off_by
    body = to_check_report(result, "Paris is the capital of France.")
    assert body["review"]["passes"]
    assert body["review"]["signed_off_by"]
    assert "changes" in body["review"]
    unused = TwoPassReview
    assert unused is not None
