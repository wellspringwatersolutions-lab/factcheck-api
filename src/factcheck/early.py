"""Stop searching a claim once the evidence already agrees."""

from __future__ import annotations

import re

from factcheck.checks import entailment
from factcheck.credibility import grade_for
from factcheck.retain import is_fact_checker, is_official
from factcheck.textutil import hostname, overlap

_RATING = re.compile(
    r"\b(false|true|mostly false|mostly true|correct|incorrect|pants on fire)\b",
    re.I,
)


def _domain(hit) -> str:
    return (getattr(hit, "origin_id", "") or hostname(getattr(hit, "url", "") or "") or getattr(hit, "url", "")).lower()


def _grade(hit) -> str:
    grade = getattr(hit, "grade", "") or ""
    if grade and grade != "E":
        return grade
    return grade_for(getattr(hit, "url", "") or "")


def _reputable(hit) -> bool:
    if getattr(hit, "provider", "") == "google_factcheck":
        return True
    if is_fact_checker(getattr(hit, "url", ""), getattr(hit, "source_type", "")) or is_official(getattr(hit, "url", "")):
        return True
    return _grade(hit) in {"A", "B"}


def _relation(claim: str, hit) -> str:
    labeled = getattr(hit, "entailment", "") or ""
    if labeled in {"entails", "contradicts"}:
        return labeled
    passage = getattr(hit, "page_text", "") or getattr(hit, "snippet", "") or ""
    return entailment(claim, passage)


def factcheck_short_circuit(claim: str, hits: list) -> bool:
    """One fact-checker rating that is about this claim is enough to stop searching."""
    for hit in hits:
        checker = getattr(hit, "provider", "") == "google_factcheck" or is_fact_checker(
            getattr(hit, "url", ""), getattr(hit, "source_type", "")
        )
        if not checker:
            continue
        blob = " ".join(
            part for part in (getattr(hit, "title", ""), getattr(hit, "snippet", ""), getattr(hit, "page_text", "")) if part
        )
        if not _RATING.search(blob):
            continue
        if overlap(claim, blob) >= 0.2 or _relation(claim, hit) in {"entails", "contradicts"}:
            return True
    return False


def search_is_enough(claim: str, hits: list) -> bool:
    """Two independent reputable sources that agree, or one fact-checker rating."""
    if factcheck_short_circuit(claim, hits):
        return True
    agreed: dict[str, set[str]] = {}
    for hit in hits:
        if not _reputable(hit):
            continue
        relation = _relation(claim, hit)
        if relation not in {"entails", "contradicts"}:
            continue
        agreed.setdefault(relation, set()).add(_domain(hit))
    return any(len(domains) >= 2 for domains in agreed.values())
