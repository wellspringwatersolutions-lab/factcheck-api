import json
import threading

import httpx

from factcheck.aggregate import decide
from factcheck.checks import apply_debunk_labels
from factcheck.credibility import grade_for
from factcheck.extract import heuristic_extract, select_key_points, split_compounds
from factcheck.synopsis import complete_sentence, fallback_tldr
from factcheck.queries import heuristic_queries, news_window
from factcheck.retrievers import DesearchRetriever
from factcheck.review import TwoPassReview, run_review
from factcheck.types import Evidence
from factcheck.web_report import to_check_report

from tests.conftest import client, make_service
from tests.test_review import _result, _reviewer


def test_news_window_follows_the_claim_and_queries_are_keywords():
    october = news_window("In October 2026 France saw a supermarket refrigeration cyberattack.")
    assert october["start_date"] == "2026-09-17"
    assert october["end_date"] == "2026-11-14"
    mid = news_window("Stockpiling guidance from mid-October 2026 told households to keep a kit.")
    assert mid == october
    assert news_window("A report from 2024 described the same program.") == {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
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


def test_desearch_runs_web_and_post_search_without_a_week_window():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content.decode())
            seen.append((request.url.path, body))
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://www.reuters.com/france",
                            "title": "France hack",
                            "highlights": ["French supermarkets lost refrigeration."],
                            "published_date": "2026-10-12",
                        }
                    ]
                },
            )
        seen.append((request.url.path, dict(request.url.params)))
        return httpx.Response(
            200,
            json={"data": [{"link": "https://example.com/web", "title": "Story", "snippet": "French supermarkets lost refrigeration."}]},
        )

    retriever = DesearchRetriever("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    hits = retriever.search_web_and_news(
        ["France supermarket refrigeration October 2026"],
        window=news_window("In October 2026 France saw a supermarket refrigeration cyberattack."),
    )
    assert not any(path == "/news" for path, _payload in seen)
    posted = [body for path, body in seen if path == "/search"]
    assert posted and posted[0]["start_date"] == "2026-09-17" and posted[0]["end_date"] == "2026-11-14"
    assert "time" not in posted[0]
    dated = next(hit for hit in hits if hit.published_at == "2026-10-12")
    assert dated.url == "https://www.reuters.com/france"
    assert "refrigeration" in dated.snippet


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


def test_a_debunk_supports_a_denial_and_a_rumor_echo_is_not_a_conflict():
    claim = "The U.S. government has not told civilians to hoard 90 days of food."
    snopes = Evidence(
        url="https://www.snopes.com/fact-check/food-hoard",
        title="Snopes",
        snippet="Snopes rating: False. The rumor that the government told civilians to hoard 90 days of food is false.",
        page_text="Snopes rating: False. The rumor that the government told civilians to hoard 90 days of food is false.",
        source_type="fact_checker",
        grade="B",
        origin_id="snopes.com",
        entailment="contradicts",
        used=True,
        quote="The rumor that the government told civilians to hoard 90 days of food is false.",
    )
    echo = Evidence(
        url="https://random-shop.example/rumor",
        title="Rumor",
        snippet="The government told civilians to hoard 90 days of food before the outage.",
        page_text="The government told civilians to hoard 90 days of food before the outage.",
        quote="The government told civilians to hoard 90 days of food before the outage.",
        grade="E",
        source_type="secondary",
        origin_id="random-shop.example",
        entailment="contradicts",
        used=True,
    )
    apply_debunk_labels(claim, [snopes, echo])
    assert snopes.entailment == "entails"
    assert echo.entailment != "contradicts"
    decision = decide([snopes, echo], category="T6", claim_text=claim)
    assert decision.verdict != "Conflicting"
    assert decision.verdict in {"likely_true", "Supported"}


def test_three_lower_tier_sources_or_one_official_page_are_likely_true():
    quote = "The European guidance asks households to keep supplies for at least 72 hours."
    pages = []
    for index, host in enumerate(("news-a.example", "news-b.example", "news-c.example", "news-d.example")):
        pages.append(
            Evidence(
                url=f"https://{host}/kit",
                title="72 hour kit",
                snippet=quote,
                page_text=quote,
                quote=quote,
                entailment="entails",
                used=True,
                grade="C",
                source_type="secondary",
                origin_id=host,
            )
        )
    assert decide(pages, category="T6", claim_text="Households were told to keep a 72 hour emergency kit.").verdict == "likely_true"
    official = Evidence(
        url="https://www.ready.gov/kit",
        title="Ready.gov",
        snippet="Build a kit with supplies for several days, including water and food.",
        page_text="Build a kit with supplies for several days, including water and food.",
        grade="A",
        source_type="primary",
        origin_id="ready.gov",
    )
    assert decide([official], category="T6", claim_text="Official guidance says households should build a kit with water and food.").verdict == "likely_true"


def test_compound_fallback_splits_clauses_and_fragments_are_rejected():
    parts = split_compounds(
        "The council raised property taxes in 2019; the mayor opposed the increase and neighbors complained."
    )
    assert any("council raised" in part for part in parts)
    assert any("mayor opposed" in part for part in parts)
    both = split_compounds("The council raised property taxes in 2019 and the mayor opposed that increase in public.")
    assert len(both) == 2
    assert split_compounds("Research and development spending rose in 2019 after the vote.") == [
        "Research and development spending rose in 2019 after the vote."
    ]
    assert not complete_sentence(", but the U.S.")
    assert not complete_sentence("His presidency and.")
    assert complete_sentence("The checked claims are mostly accurate.")
    assert fallback_tldr("Mostly accurate", "Mostly accurate. , but the U.S.").startswith("The checked")


def test_concurrent_checks_do_not_share_usage():
    from factcheck.request_ctx import active_meter
    from tests.conftest import RecordingRetriever

    class Charging(RecordingRetriever):
        def search(self, query, *, limit=6, scientific=False, categories=None, extra_queries=None, window=None, broad=False):
            found = super().search(
                query,
                limit=limit,
                scientific=scientific,
                categories=categories,
                extra_queries=extra_queries,
                window=window,
                broad=broad,
            )
            meter = active_meter(None)
            if meter is not None:
                meter.add_tokens(model="m", provider="p", input_tokens=10, output_tokens=1, cost=0.01)
            return found

    page = "The council raised property taxes in 2019 after a long public debate."
    service, _, _, _ = make_service(
        retriever=Charging([Evidence(url="https://www.reuters.com/taxes", title="Taxes", snippet=page, page_text=page, grade="B")]),
    )

    def run(text, fingerprint, box):
        box.append(service.verify(text=text, claims=None, client=client(fingerprint=fingerprint), charge=False))

    first: list = []
    second: list = []
    left = threading.Thread(target=run, args=("The council raised property taxes in 2019 after a long debate.", "one", first))
    right = threading.Thread(target=run, args=("Canberra is the capital city of Australia and has been since 1913.", "two", second))
    left.start()
    right.start()
    left.join()
    right.join()
    assert first[0].usage.retrieval_calls >= 1
    assert second[0].usage.retrieval_calls >= 1
    assert first[0].usage.input_tokens == 10
    assert second[0].usage.input_tokens == 10
    assert first[0].verification_id != second[0].verification_id
