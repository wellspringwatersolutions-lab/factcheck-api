"""POST /check body, shaped for the public factcheck site."""

from __future__ import annotations

from datetime import datetime, timezone

from factcheck.schemas import VerificationOut
from factcheck.textutil import hostname

_SITE_CATEGORY = {
    "T1": "quantity",
    "T2": "scientific",
    "T3": "scientific",
    "T4": "scientific",
    "T5": "event",
    "T6": "credibility",
    "T7": "credibility",
    "T8": "policy",
}
_SOURCE_TYPE = {
    "primary": "primary_document",
    "textbook": "primary_document",
    "data": "primary_document",
    "systematic_review": "primary_document",
    "rct": "primary_document",
    "observational": "primary_document",
    "secondary": "secondary",
    "expert_opinion": "expert_opinion",
    "eyewitness": "eyewitness",
    "anecdote": "anecdote",
    "fact_checker": "fact_checker",
    "structured": "structured",
    "primary_source": "primary_document",
}


def _checked_on(value: str | None) -> str:
    if value and len(value) >= 10 and value[4] == "-":
        return value[:10]
    return datetime.now(timezone.utc).date().isoformat()


def _source_type(raw: str, url: str) -> str:
    host = hostname(url)
    if url.startswith("tool://"):
        return "tool_result"
    if host in {"snopes.com", "politifact.com", "factcheck.org", "reuters.com", "apnews.com"}:
        return "fact_checker"
    if "gao.gov" in host:
        return "audit_report"
    return _SOURCE_TYPE.get(raw, raw or "secondary")


def to_check_report(result: VerificationOut, input_text: str) -> dict:
    claims = []
    for claim in result.claims:
        verdict = (claim.verdict or "unverified").lower()
        if verdict == "unsupported":
            verdict = "unverified"
        sources = []
        for source in claim.sources:
            if not source.quote and not source.used:
                continue
            sources.append(
                {
                    "url": source.url,
                    "title": source.title,
                    "domain": hostname(source.url),
                    "checked_on": _checked_on(source.published_at),
                    "quote": source.quote or source.snippet,
                    "source_type": _source_type(source.source_type, source.url),
                }
            )
        if claim.tool and claim.tool.get("trace"):
            sources.append(
                {
                    "url": f"tool://{claim.tool.get('tool', 'calculator')}",
                    "title": str(claim.tool.get("tool") or "tool"),
                    "domain": "tool",
                    "checked_on": datetime.now(timezone.utc).date().isoformat(),
                    "quote": str(claim.tool["trace"]),
                    "source_type": "tool_result",
                }
            )
        mainstream = None
        if claim.mainstream_check is not None:
            mainstream = {
                "items": [
                    {
                        "publisher": item.outlet,
                        "rating_label": item.rating,
                        "reason": claim.reasoning,
                        "url": item.url,
                        "title": item.outlet,
                        "quote": item.rating,
                    }
                    for item in claim.mainstream_check.checks
                ]
            }
        needle = claim.span if claim.span and input_text and claim.span in input_text else claim.text
        start = input_text.find(needle) if input_text else -1
        category = "prediction" if claim.verdict == "Hypothesis" else _SITE_CATEGORY.get(claim.claim_category, "event")
        row = {
            "claim": claim.text,
            "claim_type": claim.claim_type,
            "claim_category": category,
            "verdict": verdict,
            "verdict_label": "Likely true" if verdict == "likely_true" else verdict.replace("_", " ").title(),
            "span": claim.span,
            "strength_of_support": claim.strength_of_support,
            "checked_at": claim.checked_at,
            "would_settle": claim.would_settle,
            "sources": sources,
            "mainstream_check": mainstream,
        }
        if start >= 0:
            row["start"] = start
            row["end"] = start + len(needle)
        claims.append(row)
    synopsis = result.synopsis
    if synopsis is None:
        synopsis_out = {
            "rating": "Not enough evidence",
            "summary": "Not enough evidence. No synopsis was stored for this check.",
            "tldr": "There wasn't enough evidence to settle this.",
            "answer": "not_enough_evidence",
            "caveats": [],
            "unchecked_count": 0,
            "counts": {
                "supported": 0,
                "contradicted": 0,
                "conflicting": 0,
                "likely_true": 0,
                "unverified": 0,
                "hypothesis": 0,
                "opinion": 0,
                "not_checked": 0,
            },
        }
    else:
        synopsis_out = {
            "rating": synopsis.rating,
            "summary": synopsis.summary,
            "tldr": synopsis.tldr,
            "answer": synopsis.answer,
            "caveats": list(synopsis.caveats),
            "unchecked_count": synopsis.unchecked_count,
            "counts": synopsis.counts.model_dump(),
        }
    draft = None
    if result.draft_answer is not None:
        draft = {
            "model": result.draft_answer.model,
            "text": result.draft_answer.text,
            "stance": result.draft_answer.stance,
            "correction": result.draft_answer.correction,
        }
    review = result.review
    review_out = {
        "passes": list(review.passes),
        "changes": [
            {"claim": item.claim, "from": item.from_, "to": item.to, "reason": item.reason} for item in review.changes
        ],
        "signed_off_by": review.signed_off_by,
    }
    return {
        "input_text": input_text,
        "claims": claims,
        "mode": result.mode,
        "draft_answer": draft,
        "synopsis": synopsis_out,
        "review": review_out,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "estimated_cost_usd": result.usage.estimated_cost_usd,
            "retriever_cost_usd": result.usage.retriever_cost_usd,
            "retrieval_calls": result.usage.retrieval_calls,
            "credits_charged": result.usage.credits_charged,
            "tier": result.usage.tier,
            "models": dict(result.usage.models),
        },
    }
