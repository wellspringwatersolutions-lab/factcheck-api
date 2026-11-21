"""Two-pass review of a finished check. The models may only cite evidence already retrieved."""

from __future__ import annotations

import time

from factcheck.checks import quote_exact
from factcheck.llm import LLMError, estimate_cost, loads_object
from factcheck.schemas import ReviewChange, ReviewOut, VerificationOut
from factcheck.synopsis import (
    _RATINGS,
    _accept_tldr,
    build_synopsis,
    grounded,
)

_CAUTION = {"Supported": 0, "Contradicted": 1, "Hypothesis": 2, "Conflicting": 3, "Unverified": 4}
_VERDICTS = {"Supported", "Contradicted", "Unverified", "Conflicting", "Hypothesis"}
_INSTRUCTION = """You review a finished fact-check. You do not add facts.
Flag a verdict that its own quote does not back.
You may propose a different verdict only by copying a verbatim quote or tool trace that is already in the evidence.
You may not use your own knowledge. If no such quote exists, leave the verdict Unverified.
Flag a tldr or summary that overstates or contradicts the claim verdicts, and propose a plainer wording using only those verdicts and quotes.
In question mode the draft's yes or no, synopsis.answer, the tldr, and the rating must all agree.
Return JSON {"corrections": [{"claim": "claim text", "to": "Supported|Contradicted|Unverified|Conflicting|Hypothesis", "quote": "verbatim quote or empty", "reason": "short"}], "tldr": "", "summary": ""}"""


def _caution(verdict: str | None) -> int:
    return _CAUTION.get(verdict or "Unverified", 4)


def _packet(result: VerificationOut) -> str:
    lines = []
    if result.draft_answer is not None:
        lines.append(f"Draft ({result.draft_answer.model}): {result.draft_answer.text}")
    if result.synopsis is not None:
        lines.append(f"Synopsis rating: {result.synopsis.rating}")
        lines.append(f"tldr: {result.synopsis.tldr}")
        lines.append(f"summary: {result.synopsis.summary}")
        lines.append("caveats: " + " | ".join(result.synopsis.caveats))
    for claim in result.claims:
        lines.append(f"CLAIM: {claim.text}")
        lines.append(f"verdict: {claim.verdict}")
        if claim.tool and claim.tool.get("trace"):
            lines.append(f"tool: {claim.tool.get('trace')}")
        for source in claim.sources:
            lines.append(f"source: {source.url}")
            lines.append(f"quote: {source.quote or source.snippet}")
    return "\n".join(lines)


def _evidence_text(claim) -> str:
    parts = [source.quote or "" for source in claim.sources]
    parts.extend(source.snippet or "" for source in claim.sources)
    if claim.tool and claim.tool.get("trace"):
        parts.append(str(claim.tool["trace"]))
    return "\n".join(part for part in parts if part)


def _find_claim(result: VerificationOut, text: str):
    wanted = " ".join((text or "").split()).lower()
    if not wanted:
        return None
    for claim in result.claims:
        current = " ".join(claim.text.split()).lower()
        if current == wanted or current.startswith(wanted) or wanted.startswith(current):
            return claim
    return None


def _cite(claim, quote: str) -> bool:
    if not quote:
        return False
    if quote_exact(_evidence_text(claim), quote):
        return True
    return False


def _attach_quote(claim, quote: str, verdict: str) -> None:
    for source in claim.sources:
        if quote_exact(source.quote or source.snippet or "", quote) or quote_exact(source.snippet or "", quote):
            source.quote = quote
            source.used = True
            source.entailment = "contradicts" if verdict == "Contradicted" else "entails"
            source.stance = "contradicts" if verdict == "Contradicted" else "supports"
            return


def apply_corrections(result: VerificationOut, proposed: dict) -> list[ReviewChange]:
    """Keep a verdict change only when a verbatim quote or tool trace already in the result backs it."""
    changes: list[ReviewChange] = []
    for item in proposed.get("corrections") or []:
        claim = _find_claim(result, str(item.get("claim") or ""))
        target = str(item.get("to") or "")
        quote = str(item.get("quote") or "").strip()
        reason = str(item.get("reason") or "").strip()[:300]
        if claim is None or claim.claim_type == "opinion" or claim.verdict == "Hypothesis":
            continue
        if target not in _VERDICTS or target == claim.verdict:
            continue
        current = claim.verdict or "Unverified"
        if target in {"Supported", "Contradicted"}:
            if not _cite(claim, quote):
                continue
            _attach_quote(claim, quote, target)
            claim.strength_of_support = max(claim.strength_of_support, 0.55)
            claim.confidence = claim.strength_of_support
        elif target == "Conflicting":
            quotes = [source.quote for source in claim.sources if source.quote]
            if len(quotes) < 2 and not _cite(claim, quote):
                continue
        claim.verdict = target
        if target == "Unverified":
            claim.strength_of_support = min(claim.strength_of_support, 0.35)
        changes.append(ReviewChange(claim=claim.text, **{"from": current, "to": target, "reason": reason or "Review correction."}))
    _refresh_wording(result, proposed)
    return changes


def _refresh_wording(result: VerificationOut, proposed: dict) -> None:
    question = result.input_summary if result.mode == "question" else None
    refreshed = build_synopsis(list(result.claims), question=question, phrasing=None)
    brief_claims = "\n".join(
        f"{claim.verdict}: {claim.text} {source.quote}" for claim in result.claims for source in claim.sources
    )
    brief = f"{refreshed.rating}\n{refreshed.summary}\n{brief_claims}"
    summary = str(proposed.get("summary") or "").strip()
    tldr = str(proposed.get("tldr") or "").strip()
    if summary and grounded(summary, brief):
        for other in _RATINGS:
            if other != refreshed.rating and summary.lower().startswith(other.lower()):
                summary = ""
                break
        if summary:
            refreshed.summary = summary if summary.lower().startswith(refreshed.rating.lower()) else f"{refreshed.rating}. {summary}"
    if tldr and _accept_tldr(tldr, brief, refreshed.rating):
        refreshed.tldr = tldr if tldr.endswith((".", "!", "?")) else tldr + "."
    result.synopsis = refreshed


def _ask(reviewer, model: str, result: VerificationOut, timeout: float) -> dict:
    response = reviewer.client.complete(
        model=model,
        messages=[
            {"role": "system", "content": _INSTRUCTION},
            {"role": "user", "content": _packet(result)},
        ],
        temperature=0.0,
        max_tokens=500,
        timeout=timeout,
    )
    if reviewer.meter is not None:
        reviewer.meter.add_tokens(
            model=response.model,
            provider=response.provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=estimate_cost(reviewer.settings, response.model, response.input_tokens, response.output_tokens),
        )
    payload = loads_object(response)
    return payload if isinstance(payload, dict) else {}


class TwoPassReview:
    """Qwen proposes corrections. DeepSeek signs off. A short clock skips both."""

    def __init__(self, client, settings, meter=None):
        self.client = client
        self.settings = settings
        self.meter = meter
        self.pass1_model = settings.judge_b_model
        self.pass2_model = settings.judge_a_model


def run_review(result: VerificationOut, *, reviewer: TwoPassReview, deadline: float | None, meter) -> None:
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining < 5:
        result.warnings = list(dict.fromkeys([*result.warnings, "review skipped"]))
        result.review = ReviewOut()
        return
    original_verdicts = {claim.text: claim.verdict for claim in result.claims}
    original_synopsis = result.synopsis.model_copy(deep=True) if result.synopsis is not None else None
    try:
        timeout = 6.0 if remaining is None else max(1.0, min(6.0, remaining - 1.0))
        first = _ask(reviewer, reviewer.pass1_model, result, timeout)
        first_changes = apply_corrections(result, first)
        after_first = {claim.text: claim.verdict for claim in result.claims}
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining < 3:
            raise TimeoutError("review skipped")
        timeout = 6.0 if remaining is None else max(1.0, min(6.0, remaining - 0.5))
        second = _ask(reviewer, reviewer.pass2_model, result, timeout)
        second_changes = apply_corrections(result, second)
    except (LLMError, TimeoutError, Exception):  # noqa: BLE001 - a late review must not fail the check
        for claim in result.claims:
            claim.verdict = original_verdicts.get(claim.text, claim.verdict)
        result.synopsis = original_synopsis
        result.warnings = list(dict.fromkeys([*result.warnings, "review skipped"]))
        result.review = ReviewOut()
        _sync_usage(result, meter)
        return
    disagreed = False
    for claim in result.claims:
        if claim.claim_type == "opinion" or claim.verdict == "Hypothesis":
            continue
        earlier = after_first.get(claim.text)
        if earlier != claim.verdict:
            cautious = earlier if _caution(earlier) >= _caution(claim.verdict) else claim.verdict
            if cautious != claim.verdict:
                claim.verdict = cautious
            disagreed = True
    if disagreed and result.synopsis is not None:
        note = "The two review passes disagreed. The more cautious verdict was kept."
        if note not in result.synopsis.caveats:
            result.synopsis.caveats.append(note)
        result.synopsis = build_synopsis(list(result.claims), question=result.input_summary if result.mode == "question" else None, phrasing=None)
        if note not in result.synopsis.caveats:
            result.synopsis.caveats.append(note)
    reasons = {item.claim: item.reason for item in [*first_changes, *second_changes]}
    changes = [
        ReviewChange(claim=claim.text, **{"from": original_verdicts.get(claim.text) or "", "to": claim.verdict or "", "reason": reasons.get(claim.text, "Review correction.")})
        for claim in result.claims
        if original_verdicts.get(claim.text) != claim.verdict
    ]
    result.review = ReviewOut(passes=[reviewer.pass1_model, reviewer.pass2_model], changes=changes, signed_off_by=reviewer.pass2_model)
    from factcheck.reconcile import reconcile_answer

    reconcile_answer(result)
    _sync_usage(result, meter)


def _sync_usage(result: VerificationOut, meter) -> None:
    if meter is None:
        return
    result.usage.input_tokens = meter.input_tokens
    result.usage.output_tokens = meter.output_tokens
    result.usage.estimated_cost_usd = round(meter.estimated_cost_usd, 6)
    result.usage.models = dict(meter.models)
