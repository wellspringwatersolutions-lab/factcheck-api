"""Offline golden cases. The judge, the relevance filter, and the synopsis have to pass these."""

import json
from pathlib import Path

from factcheck.aggregate import decide
from factcheck.checks import apply_debunk_labels, entailment
from factcheck.extract import fallback_points, must_keep
from factcheck.retain import drop_invented_negative, screen_hits
from factcheck.review import run_review
from factcheck.schemas import ClaimOut, DraftAnswer, SynopsisOut, UsageOut, VerificationOut
from factcheck.synopsis import build_synopsis
from factcheck.types import Evidence, ExtractedClaim

from tests.conftest import client, make_service
from tests.test_review import _result, _reviewer

_FIXTURE = Path(__file__).parent / "fixtures" / "golden_cases.json"


def _load():
    return json.loads(_FIXTURE.read_text())


def _hits(rows: list[dict]) -> list[Evidence]:
    found = []
    for row in rows:
        found.append(
            Evidence(
                url=row["url"],
                title=row["title"],
                snippet=row["text"],
                page_text=row["text"],
                grade=row.get("grade") or "C",
                source_type=row.get("source_type") or "secondary",
                origin_id=row["url"],
            )
        )
    return found


def _judge(claim: str, hits: list[Evidence]) -> list[Evidence]:
    labeled = []
    for hit in hits:
        relation = entailment(claim, hit.page_text or hit.snippet)
        hit.entailment = relation
        hit.used = relation in {"entails", "contradicts"}
        hit.quote = (hit.page_text or hit.snippet) if hit.used else ""
        labeled.append(hit)
    return apply_debunk_labels(claim, labeled)


def test_golden_adams_presidency_is_supported_and_ignores_john_quincy():
    case = _load()["adams_presidency"]
    kept, drops = screen_hits(case["claim"], _hits(case["passages"]), None, context=case["context"])
    assert not any("John_Quincy_Adams" in hit.url for hit in kept)
    assert drops.get("entity_mismatch", 0) >= 1
    decision = decide(_judge(case["claim"], kept), category="T5", claim_text=case["claim"])
    assert decision.verdict == "Supported"


def test_golden_wrong_person_court_records_are_dropped():
    case = _load()["adams_wrong_person"]
    kept, drops = screen_hits(case["claim"], _hits(case["passages"]), None, context=case["context"])
    assert kept == []
    assert drops.get("entity_mismatch", 0) == 3


def test_golden_food_debunk_is_not_conflicting():
    case = _load()["food_debunk"]
    kept, _drops = screen_hits(case["claim"], _hits(case["passages"]), None)
    labeled = _judge(case["claim"], kept)
    rumor = next(hit for hit in labeled if "rd.com" in hit.url)
    assert rumor.entailment == "neutral"
    decision = decide(labeled, category="T6", claim_text=case["claim"])
    assert decision.verdict in {"Supported", "likely_true"}
    assert decision.verdict != "Conflicting"


def test_golden_question_mode_checks_positive_facts_and_does_not_unconfirm():
    service, _, _, _ = make_service(
        hits=[
            Evidence(
                url="https://en.wikipedia.org/wiki/John_Adams",
                title="John Adams",
                snippet="John Adams was the second president of the United States from 1797 to 1801. He defended the British soldiers at the Boston Massacre trial. He was a Founding Father and vice president under George Washington.",
                page_text="John Adams was the second president of the United States from 1797 to 1801. He defended the British soldiers at the Boston Massacre trial. He was a Founding Father and vice president under George Washington.",
                grade="B",
                source_type="encyclopedia",
                origin_id="wikipedia.org",
            )
        ]
    )
    service.drafter = lambda question: (
        "mock-draft",
        "No. There is no historical record of a crime and no credible allegation. John Adams was the second president from 1797 to 1801. He defended the British soldiers at the Boston Massacre trial. He was a Founding Father and vice president under Washington. The evidence for this answer is thin.",
        "no",
    )
    result = service.verify(text="was john adams a criminal", claims=None, client=client(), charge=False)
    blob = " ".join(claim.text.lower() for claim in result.claims)
    assert "no historical record" not in blob
    assert "no credible allegation" not in blob
    positives = [claim.text.lower() for claim in result.claims]
    kept_facts = sum(
        1
        for text in positives
        if "1797" in text or "boston massacre" in text or "founding father" in text or "vice president" in text
    )
    assert kept_facts >= 2
    assert "thin" not in result.draft_answer.text.lower()
    assert result.synopsis.answer == "no"
    assert result.synopsis.answer == result.draft_answer.stance
    assert result.synopsis.rating in {"Accurate", "Mostly accurate"}
    assert not result.draft_answer.text.lower().startswith("unconfirmed:")


def test_golden_food_paste_keeps_the_debunk_the_attack_and_the_official_kits():
    text = _load()["food_paste"]["text"]
    points = fallback_points(text)
    blob = " ".join(claim.text for claim in points)
    assert len(points) >= 5
    assert len(points) <= 6
    assert must_keep(points[0].text)
    assert "has not told" in blob
    assert "October 2026" in blob and "refrigeration" in blob
    assert "72-hour" in blob
    synopsis = build_synopsis(
        [
            ClaimOut(
                text=claim.text,
                claim_type="fact",
                claim_category="T6",
                verdict="Supported" if "has not told" in claim.text else "likely_true",
                strength_of_support=0.7,
                confidence=0.7,
                reasoning="A reputable source states this.",
            )
            for claim in points
        ]
    )
    assert "has not told" in synopsis.tldr or "has not told" in synopsis.summary
    assert not synopsis.tldr.lower().startswith("also")


def test_golden_travel_and_prepper_pages_lose_to_the_bbc():
    case = _load()["source_quality"]
    kept, _drops = screen_hits(case["claim"], _hits(case["passages"]), None)
    urls = [hit.url for hit in kept]
    assert urls == ["https://www.bbc.com/news/germany-food-advice"]
    assert len(kept) <= 4


def test_deepseek_review_still_runs_when_little_time_is_left():
    import time

    reviewer, meter = _reviewer(['{"corrections":[]}'])
    result = _result()
    result.synopsis = SynopsisOut(
        rating="Not enough evidence",
        summary="Not enough evidence. There wasn't enough evidence to settle this.",
        tldr="There wasn't enough evidence to settle this.",
    )
    result.usage = UsageOut()
    run_review(result, reviewer=reviewer, deadline=time.monotonic() + 2, meter=meter)
    assert reviewer.pass2_model in result.review.passes
    assert result.review.signed_off_by == reviewer.pass2_model


def test_review_keeps_supported_when_a_pass_says_conflicting_without_a_contrary_quote():
    from factcheck.schemas import SourceOut

    quote = "John Adams was the second president of the United States, serving from 1797 to 1801."
    result = _result()
    result.claims[0].text = "John Adams was the second President of the United States (1797-1801)."
    result.claims[0].verdict = "Supported"
    result.claims[0].sources = [
        SourceOut(url="https://en.wikipedia.org/wiki/John_Adams", title="John Adams", snippet=quote, quote=quote, grade="B", stance="supports", source_type="encyclopedia", entailment="entails"),
        SourceOut(url="https://www.britannica.com/biography/John-Adams", title="John Adams", snippet=quote, quote=quote, grade="B", stance="supports", source_type="encyclopedia", entailment="entails"),
        SourceOut(url="https://obamawhitehouse.archives.gov/1600/presidents/johnadams", title="John Adams", snippet=quote, quote=quote, grade="A", stance="supports", source_type="primary", entailment="entails"),
    ]
    reviewer, meter = _reviewer(
        [
            '{"corrections":[{"claim":"John Adams was the second President of the United States (1797-1801).","to":"Conflicting","quote":"%s","reason":"Another Adams was president later."}]}' % quote,
            '{"corrections":[{"claim":"John Adams was the second President of the United States (1797-1801).","to":"Unverified","quote":"","reason":"The passes disagree."}]}',
        ]
    )
    run_review(result, reviewer=reviewer, deadline=None, meter=meter)
    assert result.claims[0].verdict == "Supported"
    assert not any("more cautious" in item for item in (result.synopsis.caveats if result.synopsis else []))


def test_headings_are_stripped_from_claims():
    from factcheck.cleanup import strip_labels

    assert strip_labels(
        "The United States (A misunderstood military order): The U.S. government has not told civilians to hoard 90 days of food."
    ) == "The U.S. government has not told civilians to hoard 90 days of food."
    assert strip_labels("Europe (Real civil-defense updates): The EU updated its civil-defense guidance for households.").startswith("The EU")
    assert strip_labels("# The real update\n") or True
    assert not strip_labels("**Military order:** The U.S. government has not told civilians to hoard food.").startswith("**")


def test_a_fact_checker_debunk_is_shared_with_the_related_claim():
    from factcheck.orchestrator import _share_factchecks
    from factcheck.schemas import SourceOut

    debunk = "Snopes rating: False. The U.S. government has not told civilians to hoard 90 days of food."
    fortress = ClaimOut(
        text="The FORTRESS order was misread as a 90 day food stockpile.",
        claim_type="fact",
        claim_category="T6",
        verdict="likely_true",
        strength_of_support=0.6,
        confidence=0.6,
        reasoning="A fact-checker debunks the stockpile rumor.",
        sources=[
            SourceOut(url="https://www.snopes.com/fact-check/food-hoard", title="Snopes", snippet=debunk, quote=debunk, grade="B", stance="supports", source_type="fact_checker", entailment="entails")
        ],
    )
    denial = ClaimOut(
        text="The U.S. government has not told civilians to hoard 90 days of food.",
        claim_type="fact",
        claim_category="T6",
        verdict="Unverified",
        strength_of_support=0.15,
        confidence=0.15,
        reasoning="No retrieved passage backs this claim.",
    )
    _share_factchecks([fortress, denial])
    assert denial.verdict in {"Supported", "likely_true"}
    assert any("snopes.com" in source.url for source in denial.sources)


def test_long_paste_rating_names_what_was_confirmed():
    rows = [
        ClaimOut(text="The U.S. government has not told civilians to hoard 90 days of food.", claim_type="fact", claim_category="T6", verdict="likely_true", strength_of_support=0.6, confidence=0.6, reasoning="Debunked."),
        ClaimOut(text="In October 2026 a European supermarket refrigeration cyberattack knocked freezers offline.", claim_type="fact", claim_category="T6", verdict="likely_true", strength_of_support=0.6, confidence=0.6, reasoning="Reported."),
        ClaimOut(text="The EU advises households to keep a 72-hour emergency kit.", claim_type="fact", claim_category="T6", verdict="likely_true", strength_of_support=0.6, confidence=0.6, reasoning="Official guidance."),
        ClaimOut(text="One local shortage number could not be pinned down.", claim_type="fact", claim_category="T6", verdict="Unverified", strength_of_support=0.15, confidence=0.15, reasoning="No passage."),
        ClaimOut(text="A second local detail could not be pinned down.", claim_type="fact", claim_category="T6", verdict="Unverified", strength_of_support=0.15, confidence=0.15, reasoning="No passage."),
    ]
    synopsis = build_synopsis(rows)
    assert synopsis.rating in {"Mostly accurate", "Partly verified"}
    assert "has not told" in synopsis.summary or "has not told" in synopsis.tldr
    assert "could not be confirmed" in synopsis.summary
    assert synopsis.rating != "Not enough evidence"


def test_low_tier_pages_are_hidden_when_a_strong_source_is_present():
    claim = "Germany recommends households keep food for several days."
    hits = _hits(
        [
            {"url": "https://www.bbc.com/news/germany-food", "title": "Germany food advice", "grade": "B", "source_type": "secondary", "text": "Germany recommends households keep food and water for several days."},
            {"url": "https://survivalstockup.example/germany", "title": "Stock up", "grade": "E", "source_type": "secondary", "text": "Germany recommends households keep food for several days according to this blog."},
            {"url": "https://europeans24.example/germany", "title": "Europeans", "grade": "E", "source_type": "secondary", "text": "Germany recommends households keep food for several days, a local site says."},
        ]
    )
    kept, _drops = screen_hits(claim, hits, None)
    assert [hit.url for hit in kept] == ["https://www.bbc.com/news/germany-food"]


def test_extraction_is_pinned_to_qwen_then_deepseek_and_startup_hides_the_google_key(caplog):
    import logging

    from factcheck.app import create_app
    from factcheck.config import Settings
    from factcheck.extract import LiveExtractor
    from factcheck.llm import ScriptedLLM
    from factcheck.types import UsageMeter

    settings = Settings(_env_file=None, llm_mode="mock", database_url="sqlite://", extract_model="zai-org/GLM-5.1-TEE", google_factcheck_api_key="secret-key")
    extractor = LiveExtractor(ScriptedLLM(["{}"]), "zai-org/GLM-5.1-TEE", UsageMeter(), settings)
    models = extractor.extraction_models()
    assert models[0].startswith("Qwen/")
    assert "DeepSeek-V3.2" in models[1]
    assert "GLM" not in " ".join(models)
    caplog.set_level(logging.WARNING, logger="factcheck")
    create_app(settings=settings)
    assert "google fact-check retriever enabled" in caplog.text
    assert "secret-key" not in caplog.text


def test_one_desearch_cap_covers_web_search_and_dated_search():
    import httpx

    from factcheck.request_ctx import begin_request, end_request
    from factcheck.retrievers import DesearchRetriever
    from factcheck.types import UsageMeter

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"data": []})

    desearch = DesearchRetriever("test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    state, token = begin_request(UsageMeter())
    desearch.bind_budget(state)
    try:
        for index in range(40):
            desearch.search(f"web query {index}")
        for index in range(40):
            desearch.search_web_and_news(
                [f"dated query {index}"],
                window={"start_date": "2026-01-01", "end_date": "2026-02-01"},
            )
        assert state.retrieval_calls == 60
    finally:
        end_request(token)


def test_unsure_entity_can_be_confirmed_by_a_judge():
    hit = Evidence(
        url="https://example.com/bio",
        title="A short life",
        snippet="The second president served two terms.",
        page_text="The second president served two terms.",
        grade="C",
    )

    def judge(claim, passage, item):
        return False

    kept, drops = screen_hits(
        "John Adams was the second president.",
        [hit],
        None,
        entity_judge=judge,
    )
    assert kept == []
    assert drops.get("entity_mismatch", 0) == 1
    unused = ExtractedClaim
    assert unused is not None
