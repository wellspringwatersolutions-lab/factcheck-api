import time

from factcheck.llm import ScriptedLLM
from factcheck.review import TwoPassReview, run_review
from factcheck.schemas import ClaimOut, SourceOut, SynopsisOut, UsageOut, VerificationOut
from factcheck.types import UsageMeter
from factcheck.web_report import to_check_report

from tests.conftest import settings

_QUOTE = "Paris is the capital of France according to the national archive catalog."


def _result() -> VerificationOut:
    return VerificationOut(
        verification_id="review-1",
        created_at="2026-11-21T00:00:00+00:00",
        input_summary="Paris",
        claims=[
            ClaimOut(
                text="Paris is the capital of France.",
                claim_type="fact",
                claim_category="T6",
                verdict="Unverified",
                strength_of_support=0.15,
                confidence=0.15,
                reasoning="A passage was found, but it is not enough to confirm the claim.",
                sources=[
                    SourceOut(
                        url="https://www.archives.gov/paris",
                        title="National archive",
                        snippet=_QUOTE,
                        grade="A",
                        stance="not_relevant",
                        source_type="primary",
                    )
                ],
            )
        ],
        summary="1 fact claim: 1 Unverified.",
        attribution="Evidence summarized.",
        disclaimer="Evidence summarized.",
        usage=UsageOut(),
        upgrade_url="http://127.0.0.1:8741/upgrade",
        synopsis=SynopsisOut(
            rating="Not enough evidence",
            summary="Not enough evidence. Unverified: Paris is the capital of France.",
            tldr="There wasn't enough evidence to settle this.",
        ),
    )


def _reviewer(responses: list[str]) -> tuple[TwoPassReview, UsageMeter]:
    meter = UsageMeter()
    cfg = settings()
    return TwoPassReview(ScriptedLLM(responses), cfg, meter), meter


def test_review_upgrades_only_with_a_verbatim_quote_and_records_cost():
    reviewer, meter = _reviewer(
        [
            '{"corrections":[{"claim":"Paris is the capital of France.","to":"Supported","quote":"%s","reason":"The archive quote states it."}],"tldr":"He was pardoned in 1799."}'
            % _QUOTE,
            '{"corrections":[]}',
        ]
    )
    result = _result()
    run_review(result, reviewer=reviewer, deadline=None, meter=meter)
    assert result.claims[0].verdict == "Supported"
    assert result.claims[0].sources[0].quote == _QUOTE
    assert result.review.passes == [reviewer.pass1_model, reviewer.pass2_model]
    assert result.review.signed_off_by == reviewer.pass2_model
    assert result.review.changes[0].from_ == "Unverified"
    assert result.review.changes[0].to == "Supported"
    assert "1799" not in result.synopsis.tldr
    assert result.usage.input_tokens > 0
    assert result.usage.estimated_cost_usd > 0
    body = to_check_report(result, "Paris")
    assert body["review"]["changes"][0]["from"] == "Unverified"
    assert body["review"]["signed_off_by"] == reviewer.pass2_model


def test_review_rejects_a_quote_that_is_not_in_the_evidence():
    reviewer, meter = _reviewer(
        [
            '{"corrections":[{"claim":"Paris is the capital of France.","to":"Supported","quote":"Paris was founded on the moon.","reason":"I know this."}]}',
            '{"corrections":[]}',
        ]
    )
    result = _result()
    run_review(result, reviewer=reviewer, deadline=None, meter=meter)
    assert result.claims[0].verdict == "Unverified"
    assert result.review.changes == []
    assert result.review.signed_off_by == reviewer.pass2_model


def test_disagreement_keeps_the_more_cautious_verdict():
    reviewer, meter = _reviewer(
        [
            '{"corrections":[{"claim":"Paris is the capital of France.","to":"Supported","quote":"%s","reason":"Archive quote."}]}'
            % _QUOTE,
            '{"corrections":[{"claim":"Paris is the capital of France.","to":"Unverified","quote":"","reason":"One archive page is not enough."}]}',
        ]
    )
    result = _result()
    run_review(result, reviewer=reviewer, deadline=None, meter=meter)
    assert result.claims[0].verdict == "Unverified"
    assert not any("more cautious" in item for item in result.synopsis.caveats)


def test_review_is_skipped_when_the_budget_is_gone():
    class Boom:
        def complete(self, **kwargs):
            raise AssertionError("the review must not call a model")

    reviewer = TwoPassReview(Boom(), settings(), UsageMeter())
    result = _result()
    run_review(result, reviewer=reviewer, deadline=time.monotonic() - 1, meter=reviewer.meter)
    assert result.claims[0].verdict == "Unverified"
    assert "review skipped" in result.warnings
    assert result.review.passes == []
    assert result.review.signed_off_by == ""
