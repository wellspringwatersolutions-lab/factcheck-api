import time

from fastapi.testclient import TestClient

from factcheck.app import create_app
from factcheck.cleanup import is_bare_question, split_dialogue
from factcheck.extract import heuristic_extract
from factcheck.llm import ScriptedLLM
from factcheck.questions import AnswerDrafter, DraftError
from factcheck.report_html import render_report
from factcheck.schemas import ClaimOut
from factcheck.synopsis import build_synopsis
from factcheck.types import Evidence, UsageMeter

from tests.conftest import client, make_service, settings


def _claim(text, verdict, strength=0.8, effort="standard", reasoning="Checked against a cited passage."):
    return ClaimOut(
        text=text,
        claim_type="fact",
        claim_category="T5",
        verdict=verdict,
        strength_of_support=strength,
        confidence=strength,
        reasoning=reasoning,
        effort=effort,
    )


WASHINGTON = [
    _claim("He was never charged or convicted of a crime.", "Supported", 0.8),
    _claim("Slavery was legal in Virginia during his lifetime.", "Supported", 0.7),
    _claim("Technically he committed treason under British law.", "Supported", 0.7),
]


def test_synopsis_shape_includes_the_caveat():
    synopsis = build_synopsis(WASHINGTON, question="Was George Washington a criminal?")
    assert set(synopsis.model_dump()) == {"rating", "answer", "summary", "tldr", "caveats", "unchecked_count", "counts"}
    assert synopsis.answer == "no"
    assert synopsis.rating == "Mostly accurate"
    assert synopsis.summary.startswith("Mostly accurate")
    assert "never charged" in synopsis.summary.lower()
    assert "treason" in synopsis.summary.lower()
    assert synopsis.tldr
    assert len(synopsis.tldr.split()) <= 30
    assert "never charged" in synopsis.tldr.lower() or "treason" in synopsis.tldr.lower()
    assert synopsis.caveats
    assert synopsis.unchecked_count == 0
    assert synopsis.counts.supported == 3
    assert synopsis.counts.model_dump()["contradicted"] == 0


def test_synopsis_falls_back_when_phrasing_fails_or_invents_a_fact():
    def fail(brief, rating):
        raise RuntimeError("timeout")

    fallback = build_synopsis(WASHINGTON, question="Was George Washington a criminal?", phrasing=fail)
    assert fallback.rating == "Mostly accurate"
    assert "1799" not in fallback.summary
    assert fallback.tldr
    assert len(fallback.tldr.split()) <= 30
    assert "1799" not in fallback.tldr

    def invent(brief, rating):
        return {
            "summary": "Mostly accurate. He was pardoned in 1799 after a secret trial.",
            "tldr": "He was pardoned in 1799 after a secret trial.",
        }

    safe = build_synopsis(WASHINGTON, question="Was George Washington a criminal?", phrasing=invent)
    assert "1799" not in safe.summary
    assert "1799" not in safe.tldr
    assert "pardoned" not in safe.summary.lower()
    assert "pardoned" not in safe.tldr.lower()
    assert safe.summary.startswith("Mostly accurate")
    assert safe.tldr
    assert len(safe.tldr.split()) <= 30

    def phrase(brief, rating):
        return {
            "summary": "Mostly accurate. He was never charged or convicted of a crime, but technically he committed treason under British law.",
            "tldr": "He was never charged or convicted of a crime, but technically he committed treason under British law.",
        }

    kept = build_synopsis(WASHINGTON, question="Was George Washington a criminal?", phrasing=phrase)
    assert kept.tldr.lower().startswith("he was never charged")
    assert "1799" not in kept.tldr

    def forbidden(brief, rating):
        raise AssertionError("phrasing should not run once the budget is gone")

    skipped = build_synopsis(
        WASHINGTON,
        question="Was George Washington a criminal?",
        phrasing=forbidden,
        deadline=time.monotonic() - 1,
    )
    assert skipped.rating == "Mostly accurate"
    assert skipped.tldr
    assert len(skipped.tldr.split()) <= 30


def test_counts_put_each_claim_in_one_bucket():
    rows = [
        _claim("The first claim holds.", "Supported"),
        _claim("The second claim does not hold.", "Contradicted"),
        _claim("The sources disagree on this claim.", "Conflicting"),
        _claim("Nothing was retrieved for this claim.", "Unverified", 0.0, reasoning="No retrieved passage or tool result backs this claim."),
        ClaimOut(
            text="A new treatment might work next year.",
            claim_type="fact",
            claim_category="T8",
            verdict="Hypothesis",
            strength_of_support=0.2,
            confidence=0.2,
            reasoning="This is a hypothesis.",
        ),
        ClaimOut(
            text="The bill is a disaster.",
            claim_type="opinion",
            claim_category="T8",
            verdict=None,
            strength_of_support=0.0,
            confidence=0.0,
            reasoning="Opinion.",
        ),
        _claim(
            "This one was left out.",
            "Unverified",
            0.0,
            effort="skipped",
            reasoning="Not checked in this request.",
        ),
    ]
    counts = build_synopsis(rows).counts
    assert counts.model_dump() == {
        "supported": 1,
        "likely_true": 0,
        "contradicted": 1,
        "conflicting": 1,
        "unverified": 1,
        "hypothesis": 1,
        "opinion": 1,
        "not_checked": 1,
    }


def test_multi_turn_questions_are_context_and_the_last_answer_is_checked():
    chat = """User: Was George Washington a criminal?
Assistant: He was never charged or convicted of a crime. Slavery was legal in Virginia during his lifetime.
User: What about the Whiskey Rebellion?
Assistant: Washington led federal troops against the Whiskey Rebellion in 1794.
"""
    dialogue = split_dialogue(chat)
    assert dialogue.question.lower().startswith("what about")
    assert "1794" in dialogue.answer
    claims = heuristic_extract(chat, None)
    blob = " ".join(claim.text for claim in claims)
    assert "1794" in blob
    assert "criminal" not in blob.lower()
    assert "Slavery" not in blob
    assert all("?" not in claim.text for claim in claims)


def test_mostly_unverified_stays_cautious():
    rows = [
        _claim(f"Statement {index} is settled.", "Unverified", 0.0, reasoning="No retrieved passage or tool result backs this claim.")
        for index in range(4)
    ]
    rows.append(_claim("One weak claim is true.", "Supported", 0.2))
    rows.append(
        _claim(
            "An extra claim was left out.",
            "Unverified",
            0.0,
            effort="skipped",
            reasoning="Not checked in this request. The check focused on the highest-stakes claims.",
        )
    )
    synopsis = build_synopsis(rows)
    assert synopsis.rating == "Not enough evidence"
    assert synopsis.summary.lower().startswith("not enough evidence")
    assert "unverified" in synopsis.summary.lower()
    assert "mostly accurate" not in synopsis.summary.lower()
    assert synopsis.unchecked_count == 1
    assert synopsis.tldr == "There wasn't enough evidence to settle this."
    assert synopsis.counts.unverified == 4
    assert synopsis.counts.supported == 1
    assert synopsis.counts.not_checked == 1
    assert synopsis.counts.contradicted == 0


def test_bare_question_detection():
    assert is_bare_question("Was George Washington a criminal?")
    assert is_bare_question("was George Washington a criminal")
    assert is_bare_question("How tall is Mount Everest?")
    assert not is_bare_question("George Washington was a criminal.")
    assert not is_bare_question("garlic lowers blood pressure")
    assert not is_bare_question("He was never charged. Was he a criminal?")
    assert not is_bare_question("17 * 24 = 408")
    assert not is_bare_question("Hi?")
    assert not is_bare_question("What?")


def test_drafter_parses_the_json_answer():
    scripted = ScriptedLLM(['{"answer": "Canberra is the capital of Australia."}'])
    drafter = AnswerDrafter(scripted, "deepseek-ai/DeepSeek-V3.2-TEE", UsageMeter(), settings())
    model, text, stance = drafter("What is the capital of Australia?")
    assert text == "Canberra is the capital of Australia."
    assert stance == "not_enough_evidence"
    assert model
    broken = AnswerDrafter(ScriptedLLM(["not json"]), "deepseek-ai/DeepSeek-V3.2-TEE", UsageMeter(), settings())
    try:
        broken("What is the capital of Australia?")
    except DraftError as exc:
        assert exc.body()["error"]["code"] == "draft_failed"
    else:
        raise AssertionError("a bad draft should fail cleanly")


def test_draft_is_not_evidence_when_nothing_is_retrieved():
    service, retriever, _, _ = make_service(hits=[])
    service.drafter = lambda question: ("mock-draft", "The Eiffel Tower is in Berlin.")
    result = service.verify(
        text=None,
        claims=None,
        question="Where is the Eiffel Tower?",
        client=client(),
        charge=False,
    )
    assert result.mode == "question"
    assert result.draft_answer.model == "mock-draft"
    assert result.draft_answer.text.startswith("Unconfirmed:")
    assert "Berlin" in result.draft_answer.text
    assert result.synopsis.answer == "not_enough_evidence"
    assert result.draft_answer.stance == "not_enough_evidence"
    assert any("Berlin" in claim.text for claim in result.claims)
    assert all("?" not in claim.text for claim in result.claims)
    assert all(claim.verdict != "Supported" for claim in result.claims)
    assert all(claim.verdict == "Unverified" for claim in result.claims)
    assert result.synopsis.rating == "Not enough evidence"
    for claim in result.claims:
        for source in claim.sources:
            assert source.quote != result.draft_answer.text
    assert retriever.queries


def test_contradiction_synopsis_states_the_cited_correction():
    url = "https://www.example.edu/everest"
    passage = "Mount Everest is 8849 m tall."
    hit = Evidence(url=url, title="Survey", snippet=passage, page_text=passage, grade="A", provider="desearch")
    service, _, _, _ = make_service(hits=[hit], pages={url: (200, passage)})
    service.drafter = lambda question: ("mock-draft", "Mount Everest is 6500 m tall.")
    result = service.verify(text="How tall is Mount Everest?", claims=None, client=client(), charge=False)
    assert result.mode == "question"
    assert result.draft_answer.text.lower().startswith("correction:")
    assert "6500" in result.draft_answer.text
    assert result.draft_answer.correction
    assert result.claims[0].verdict == "Contradicted"
    assert result.synopsis.answer == "partly"
    assert result.synopsis.rating == "Mixed"
    assert "8849" in result.synopsis.summary
    assert "cited evidence" in result.synopsis.summary.lower()
    assert "6500" not in result.synopsis.summary
    quotes = [source.quote for claim in result.claims for source in claim.sources]
    assert any("8849" in quote for quote in quotes)
    assert all(source.provider != "draft" for claim in result.claims for source in claim.sources)
    page = render_report(result)
    assert page.find("Veracity synopsis") > page.find("6500")
    assert "8849" in page


def test_failed_draft_returns_503_and_does_not_spend_the_free_check():
    service, _, _, cfg = make_service()

    def fail(question):
        raise RuntimeError("chutes down")

    service.drafter = fail
    app = create_app(settings=cfg, service=service)
    headers = {"X-Client-Fingerprint": "draft-fail"}
    with TestClient(app) as http:
        failed = http.post("/check", json={"question": "Where is the Eiffel Tower?"}, headers=headers)
        assert failed.status_code == 503
        assert failed.json()["error"]["code"] == "draft_failed"
        ok = http.post("/check", json={"text": "17 * 24 = 408"}, headers=headers)
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["mode"] == "text"
        assert body["draft_answer"] is None
        assert set(body["synopsis"]) == {"rating", "answer", "summary", "tldr", "caveats", "unchecked_count", "counts"}
        assert body["synopsis"]["rating"] == "Accurate"
        assert body["synopsis"]["tldr"]
        assert body["synopsis"]["counts"]["supported"] == 1
        verify = http.post("/v1/verify", json={"text": "2 + 2 = 4"}, headers={"X-Client-Fingerprint": "draft-verify"})
        assert verify.status_code == 200
        saved = verify.json()
        assert saved["mode"] == "text"
        assert saved["draft_answer"] is None
        assert saved["synopsis"]["rating"] == "Accurate"
