"""Public verdicts. Internal states stay finer; this is the output boundary."""

from __future__ import annotations

from factcheck.schemas import VerificationOut

PUBLIC_VERDICTS = ("supported", "hypothesis", "opinion", "conflicting", "unverified")

_LABELS = {
    "supported": "Supported",
    "hypothesis": "Hypothesis",
    "opinion": "Opinion",
    "conflicting": "Conflicting",
    "unverified": "Unverified",
}

# Old and new names accepted on input. Values are the internal states the checker uses.
_INTERNAL = {
    "supported": "Supported",
    "likely_true": "likely_true",
    "hypothesis": "Hypothesis",
    "speculative": "Hypothesis",
    "opinion": "opinion",
    "conflicting": "Conflicting",
    "contradicted": "Contradicted",
    "unverified": "Unverified",
    "unsupported": "Unsupported",
}


def canonical_verdict(value: str | None) -> str | None:
    """Accept a public or legacy verdict and return the internal state."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in _INTERNAL.values() or text in {"Supported", "likely_true", "Hypothesis", "Conflicting", "Contradicted", "Unverified", "Unsupported", "opinion"}:
        return text
    return _INTERNAL.get(text.lower())


def map_verdict(verdict: str | None, claim_type: str | None = None, detail: str | None = None) -> tuple[str, str | None, str]:
    """Return public verdict, verdict_detail, and verdict_label. Idempotent."""
    if (claim_type or "") == "opinion" or (verdict or "").lower() == "opinion":
        return "opinion", None, "Opinion"
    key = (verdict or "").strip()
    lowered = key.lower()
    if lowered == "hypothesis" and detail in {"likely_true", "speculative"}:
        return "hypothesis", detail, "Hypothesis"
    if lowered == "conflicting" and detail == "contradicted":
        return "conflicting", "contradicted", "Conflicting"
    if key in {"Supported", "supported"}:
        return "supported", None, "Supported"
    if key == "likely_true":
        return "hypothesis", "likely_true", "Hypothesis"
    if key in {"Hypothesis", "hypothesis"}:
        return "hypothesis", "speculative", "Hypothesis"
    if key in {"Contradicted", "contradicted"}:
        return "conflicting", "contradicted", "Conflicting"
    if key in {"Unsupported", "unsupported", "Conflicting", "conflicting"}:
        return "conflicting", None, "Conflicting"
    return "unverified", None, "Unverified"


def _public_name(value: str | None) -> str:
    verdict, _detail, _label = map_verdict(value, "fact")
    return verdict


def confidence_and_action(verdict: str | None, claim_type: str | None, confidence: float | None) -> tuple[float, str]:
    """A score from 0 to 1 and whether an agent should state the claim."""
    try:
        score = float(confidence or 0)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    public, _detail, _label = map_verdict(verdict, claim_type)
    if public == "supported" and score >= 0.6:
        action = "safe_to_state"
    elif public in {"supported", "hypothesis"}:
        action = "state_with_caveat"
    else:
        action = "do_not_state"
    return score, action


def publish_verification(result: VerificationOut) -> dict:
    """JSON a client may see. Claim verdicts are only the five public names."""
    data = result.model_dump(mode="json", by_alias=True)
    for claim in data.get("claims") or []:
        verdict, detail, label = map_verdict(claim.get("verdict"), claim.get("claim_type"), claim.get("verdict_detail"))
        claim["verdict"] = verdict
        claim["verdict_detail"] = detail
        claim["verdict_label"] = label
        score, action = confidence_and_action(verdict, claim.get("claim_type"), claim.get("confidence"))
        claim["confidence"] = score
        claim["recommended_action"] = action
    review = data.get("review") or {}
    for change in review.get("changes") or []:
        change["from"] = _public_name(change.get("from"))
        change["to"] = _public_name(change.get("to"))
    return data
