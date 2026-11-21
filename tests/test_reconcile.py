from factcheck.reconcile import reconcile_answer
from factcheck.schemas import ClaimOut, DraftAnswer, SynopsisOut, UsageOut, VerificationOut

from tests.conftest import client, make_service


def _report(question: str, draft: str, stance: str, claims: list[ClaimOut]) -> VerificationOut:
    result = VerificationOut(
        verification_id="q",
        created_at="2026-11-21T00:00:00+00:00",
        input_summary=question,
        claims=claims,
        mode="question",
        draft_answer=DraftAnswer(model="mock-draft", text=draft, stance=stance),
        summary="",
        attribution="",
        disclaimer="",
        usage=UsageOut(),
        upgrade_url="",
        synopsis=SynopsisOut(
            rating="Not enough evidence",
            summary="Not enough evidence.",
            tldr="There wasn't enough evidence to settle this.",
            answer="not_enough_evidence",
        ),
    )
    reconcile_answer(result)
    return result


def _claim(text: str, verdict: str = "Unverified") -> ClaimOut:
    return ClaimOut(
        text=text,
        claim_type="fact",
        claim_category="T5",
        verdict=verdict,
        strength_of_support=0.7 if verdict == "Supported" else 0.15,
        confidence=0.15,
        reasoning="Checked.",
    )


def test_john_adams_no_matches_the_bottom_line():
    result = _report(
        "was John Adams a criminal",
        "No. He was never charged or convicted of a crime.",
        "no",
        [_claim("He was never charged or convicted of a crime.")],
    )
    assert result.synopsis.answer == "no"
    assert result.synopsis.rating != "Not enough evidence"
    assert result.synopsis.tldr.lower().startswith("no")
    assert result.draft_answer.stance == "no"
    assert result.draft_answer.text.lower().startswith("no")
    assert "not enough evidence" not in result.synopsis.tldr.lower()


def test_george_washington_no_keeps_the_treason_caveat():
    result = _report(
        "was George Washington a criminal",
        "No. He was never charged or convicted, but technically he committed treason under British law.",
        "no",
        [
            _claim("He was never charged or convicted.", "Supported"),
            _claim("Technically he committed treason under British law.", "Supported"),
        ],
    )
    assert result.synopsis.answer == "no"
    assert result.synopsis.rating == "Mostly accurate"
    assert result.synopsis.tldr.lower().startswith("no")
    assert "treason" in result.synopsis.tldr.lower()
    assert result.draft_answer.text.lower().startswith("no")
    assert result.draft_answer.stance == "no"


def test_checked_pipeline_keeps_adams_and_washington_consistent():
    service, _, _, _ = make_service(hits=[])
    service.drafter = lambda question: (
        "mock-draft",
        "No. He was never charged or convicted of a crime.",
        "no",
    )
    adams = service.verify(text="was John Adams a criminal", claims=None, client=client(), charge=False)
    assert adams.synopsis.answer == "no"
    assert adams.synopsis.rating != "Not enough evidence"
    assert adams.synopsis.tldr.lower().startswith("no")
    assert adams.draft_answer.text.lower().startswith("no")
    assert adams.draft_answer.stance == "no"

    service.drafter = lambda question: (
        "mock-draft",
        "No. He was never charged or convicted, but technically he committed treason under British law.",
        "no",
    )
    washington = service.verify(
        text="was George Washington a criminal",
        claims=None,
        client=client(fingerprint="gw-answer"),
        charge=False,
    )
    assert washington.synopsis.answer == "no"
    assert washington.synopsis.rating == "Mostly accurate"
    assert washington.synopsis.tldr.lower().startswith("no")
    assert "treason" in washington.synopsis.tldr.lower()
    assert washington.draft_answer.stance == "no"
