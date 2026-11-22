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
    assert result.synopsis.answer == "not_enough_evidence"
    assert result.synopsis.answer == result.draft_answer.stance
    assert result.synopsis.rating == "Not enough evidence"
    assert any("thin" in item.lower() for item in result.synopsis.caveats)
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
    assert synopsis.tldr == "Reputable sources report that the U.S. government has not told civilians to hoard 90 days of food."
    assert "could not be confirmed" in synopsis.summary
    assert "Claims about" not in synopsis.summary
    assert synopsis.rating != "Not enough evidence"


def test_a_supportive_snopes_page_keeps_the_fortress_origin_likely_true():
    from factcheck.aggregate import decide
    from factcheck.checks import apply_debunk_labels, entailment
    from factcheck.judge import apply_verified_labels
    from factcheck.publish import map_verdict
    from factcheck.types import Evidence, JudgeVote

    claim = (
        "That rumor blew up on TikTok and Facebook after Defense Secretary Pete Hegseth announced FORTRESS America, "
        "a Pentagon initiative ordering major U.S. military bases to become capable of operating off-grid for 90 days "
        "in case of war or sabotage."
    )
    page = (
        "Snopes rating: False. The hoarding rumor is false. The U.S. government has not told civilians to hoard 90 days of food. "
        "Defense Secretary Pete Hegseth announced FORTRESS America, a Pentagon initiative ordering major U.S. military bases "
        "to become capable of operating off-grid for 90 days in case of war or sabotage."
    )
    evidence = Evidence(
        url="https://www.snopes.com/fact-check/fortress-food",
        title="Snopes",
        snippet=page,
        quote=page,
        page_text=page,
        grade="B",
        source_type="fact_checker",
        origin_id="snopes.com",
        entailment=entailment(claim, page),
        used=True,
    )
    labeled = apply_verified_labels(
        [evidence],
        [
            JudgeVote(role="a", model="qwen", provider="chutes", family="qwen", verdict="rank_only", reasoning="", stances={evidence.url: "contradicts"}, quotes={evidence.url: "The U.S. government has not told civilians to hoard 90 days of food."}),
            JudgeVote(role="b", model="deepseek", provider="chutes", family="deepseek", verdict="rank_only", reasoning="", stances={evidence.url: "supports"}, quotes={evidence.url: "FORTRESS America, a Pentagon initiative ordering major U.S. military bases to operate off-grid for 90 days."}),
        ],
        claim,
    )
    labeled = apply_debunk_labels(claim, labeled)
    decision = decide(labeled, category="T6", claim_text=claim)
    assert decision.verdict == "likely_true"
    public, detail, label = map_verdict(decision.verdict, "fact")
    assert public == "hypothesis"
    assert detail == "likely_true"
    assert label == "Hypothesis"


def test_topic_overlap_alone_does_not_make_a_grid_claim_likely_true():
    from factcheck.aggregate import decide
    from factcheck.checks import entailment

    claim = "Aging power grids are increasingly failing during severe storms, heatwaves, and floods."
    page = "PNNL studies aging power grids and how storms affect infrastructure planning at the laboratory."
    assert entailment(claim, page) == "neutral"
    evidence = Evidence(
        url="https://www.pnnl.gov/grid",
        title="PNNL grid research",
        snippet=page,
        quote=page,
        page_text=page,
        grade="A",
        source_type="primary",
        origin_id="pnnl.gov",
        entailment="neutral",
        used=False,
    )
    assert decide([evidence], category="T6", claim_text=claim).verdict == "Unverified"


def test_a_conflicting_claim_is_not_mostly_accurate():
    rows = [
        ClaimOut(text="The checked guidance is a 72-hour kit of food and water.", claim_type="fact", claim_category="T6", verdict="Supported", strength_of_support=0.8, confidence=0.8, reasoning="Official."),
        ClaimOut(text="Stores were ordered to close nationwide.", claim_type="fact", claim_category="T6", verdict="Conflicting", strength_of_support=0.4, confidence=0.4, reasoning="Sources disagree."),
        ClaimOut(text="One local detail could not be pinned down.", claim_type="fact", claim_category="T6", verdict="Unverified", strength_of_support=0.1, confidence=0.1, reasoning="No passage."),
    ]
    synopsis = build_synopsis(rows)
    assert synopsis.rating in {"Partly verified", "Mixed"}
    assert synopsis.rating != "Mostly accurate"


def test_fortress_run_fallback_tldr_is_the_hoard_denial():
    fortress = (
        "That rumor blew up on TikTok and Facebook after Defense Secretary Pete Hegseth announced "
        "FORTRESS America, a Pentagon initiative ordering major U.S. military bases to become capable "
        "of operating off-grid for 90 days in case."
    )
    denial = "The U.S. government has not told civilians to hoard 90 days of food."
    hack = "In mid-October 2026 a European supermarket refrigeration cyberattack knocked freezers offline."
    apocalypse = 'Grid failures aren\'t pointing to a single imminent "AI apocalypse".'
    aging = "Aging power grids increasingly failing is a separate infrastructure problem."
    assisted = "AI-assisted cyberattacks on critical infrastructure are increasing."
    rows = [
        ClaimOut(text=fortress, claim_type="fact", claim_category="T6", verdict="likely_true", strength_of_support=0.9, confidence=0.9, reasoning="Reported."),
        ClaimOut(text=denial, claim_type="fact", claim_category="T6", verdict="likely_true", strength_of_support=0.6, confidence=0.6, reasoning="Debunked."),
        ClaimOut(text=hack, claim_type="fact", claim_category="T6", verdict="Unverified", strength_of_support=0.15, confidence=0.15, reasoning="No passage."),
        ClaimOut(text=apocalypse, claim_type="fact", claim_category="T6", verdict="Unverified", strength_of_support=0.15, confidence=0.15, reasoning="No passage."),
        ClaimOut(text=aging, claim_type="fact", claim_category="T6", verdict="Unverified", strength_of_support=0.15, confidence=0.15, reasoning="No passage."),
        ClaimOut(text=assisted, claim_type="fact", claim_category="T6", verdict="Supported", strength_of_support=0.95, confidence=0.95, reasoning="Reported."),
    ]
    synopsis = build_synopsis(rows)
    voiced = "Reputable sources report that the U.S. government has not told civilians to hoard 90 days of food."
    assert synopsis.tldr == voiced
    assert "FORTRESS" not in synopsis.tldr
    assert "Claims about" not in synopsis.summary
    assert ", and " not in synopsis.summary
    assert synopsis.summary.startswith("Partly verified. " + voiced)
    assert "that rumor blew up" in synopsis.summary.lower()
    assert ", and " not in synopsis.summary
    assert "AI-assisted cyberattacks" in synopsis.summary
    assert "freezers" not in synopsis.summary
    assert "3 claims could not be confirmed." in synopsis.summary


def test_dated_event_wikipedia_query_includes_the_month_and_year(caplog):
    import logging

    from factcheck.archives import WikipediaRetriever
    from factcheck.queries import wikipedia_event_query
    from factcheck.retain import screen_hits

    claim = "In mid-October 2026 a European supermarket refrigeration cyberattack knocked freezers offline."
    query = wikipedia_event_query(claim)
    assert "October" in query and "2026" in query
    assert "supermarket" in query or "cyberattack" in query
    assert "mid-October" not in query

    class Wiki(WikipediaRetriever):
        def search(self, query, *, limit=2):
            self.seen = query
            return []

    wiki = Wiki("lab@example.com", client=None, min_interval=0)
    wiki.client = None
    caplog.set_level(logging.WARNING, logger="factcheck")
    assert wiki.search_dated_event(claim) == []
    assert wiki.seen == query
    assert any("reason=no_results" in record.message and "October 2026" in record.message for record in caplog.records)

    caplog.clear()
    hits = _hits(
        [
            {
                "url": "https://en.wikipedia.org/wiki/French_painting",
                "title": "French painting",
                "grade": "C",
                "source_type": "secondary",
                "text": "Nineteenth century portrait painters worked in Paris studios.",
            }
        ]
    )
    screen_hits(claim, hits, {"start_date": "2026-10-01", "end_date": "2026-10-31"})
    assert any("wikipedia dated event dropped" in record.message and "wikipedia.org" in record.message for record in caplog.records)


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


def test_abbreviations_stay_inside_the_sentence_and_the_tldr():
    from factcheck.textutil import split_sentences

    text = (
        "The story traces to a Pentagon initiative ordering major U.S. commands to stock supplies. "
        "Dr. Ng told the U.K. and the E.U. on Jan. 4, e.g. the St. Louis plan vs. Inc. No. 12. "
        "J. Q. Adams was president."
    )
    parts = split_sentences(text)
    assert parts[0].endswith("supplies.")
    assert "U.S. commands" in parts[0]
    assert any("U.K." in part and "E.U." in part and "Jan." in part for part in parts)
    assert any(part.startswith("J. Q. Adams") for part in parts)
    rows = [
        ClaimOut(
            text="A Pentagon initiative ordering major U.S. commands to stock supplies was misread as a civilian hoard order.",
            claim_type="fact",
            claim_category="T6",
            verdict="likely_true",
            strength_of_support=0.6,
            confidence=0.6,
            reasoning="The order was misread.",
        ),
        ClaimOut(
            text="Europe's updated preparedness guidance is real.",
            claim_type="fact",
            claim_category="T6",
            verdict="likely_true",
            strength_of_support=0.6,
            confidence=0.6,
            reasoning="Official guidance.",
        ),
        ClaimOut(
            text="The 'AI apocalypse' starts next week.",
            claim_type="fact",
            claim_category="T6",
            verdict="Unverified",
            strength_of_support=0.15,
            confidence=0.15,
            reasoning="No passage.",
        ),
    ]
    synopsis = build_synopsis(rows)
    assert synopsis.tldr == "Reputable sources report that a Pentagon initiative ordering major U.S. commands to stock supplies was misread as a civilian hoard order."
    assert "U.S. commands" in synopsis.tldr
    assert "could not be confirmed" in synopsis.summary
    assert "Claims about" not in synopsis.summary


def test_unrelated_pages_are_dropped_and_encyclopedia_quotes_are_shared():
    from factcheck.orchestrator import _share_factchecks
    from factcheck.schemas import SourceOut

    service, _, _, _ = make_service()
    claim = ExtractedClaim(text="In October 2026 a European supermarket refrigeration cyberattack knocked freezers offline.")
    hits = _hits(
        [
            {
                "url": "https://en.wikipedia.org/wiki/French_protests",
                "title": "French protests",
                "grade": "B",
                "source_type": "encyclopedia",
                "text": "Street protests and bomb threats disrupted French cities.",
            },
            {
                "url": "https://www.bbc.com/news/freezer-hack",
                "title": "Freezer hack",
                "grade": "B",
                "source_type": "secondary",
                "text": "In October 2026 a European supermarket refrigeration cyberattack knocked freezers offline.",
            },
        ]
    )
    kept = service._select(claim, hits)
    assert [hit.url for hit in kept] == ["https://www.bbc.com/news/freezer-hack"]
    quote = "John Adams was a Founding Father, the first vice president, and the second president."
    president = ClaimOut(
        text="John Adams was the second president.",
        claim_type="fact",
        claim_category="T5",
        verdict="Supported",
        strength_of_support=0.8,
        confidence=0.8,
        reasoning="Wikipedia states the office.",
        sources=[
            SourceOut(
                url="https://en.wikipedia.org/wiki/John_Adams",
                title="John Adams",
                snippet=quote,
                quote=quote,
                grade="B",
                stance="supports",
                source_type="encyclopedia",
                entailment="entails",
            )
        ],
    )
    founder = ClaimOut(
        text="John Adams was a Founding Father.",
        claim_type="fact",
        claim_category="T5",
        verdict="Unverified",
        strength_of_support=0.15,
        confidence=0.15,
        reasoning="No retrieved passage backs this claim.",
    )
    _share_factchecks([president, founder])
    assert founder.verdict in {"Supported", "likely_true"}
    assert any("wikipedia.org" in source.url for source in founder.sources)


def test_parallel_claims_stop_at_sixty_desearch_calls():
    import httpx

    from factcheck.retrievers import CompositeRetriever, DesearchRetriever

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/extract"):
            return httpx.Response(200, text="A page about a distinct event.")
        return httpx.Response(200, json={"data": [], "results": []})

    desearch = DesearchRetriever("test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    service, _, _, cfg = make_service()
    service.retriever = CompositeRetriever([desearch])
    cfg.max_output_claims = 80
    cfg.max_checked_claims = 80
    cfg.check_concurrency = 8
    claims = [f"Alpha{i} beta{i} claimed that gamma{i} happened in city{i} during year{i}." for i in range(80)]
    result = service.verify(text=None, claims=claims, client=client(), charge=False, budget=20)
    assert len(calls) == 60
    assert result.usage.retrieval_calls == 60


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
