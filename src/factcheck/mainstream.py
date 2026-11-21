"""What mainstream fact-checkers say about opinions. Never a true or false label."""

from __future__ import annotations

import re

from factcheck.textutil import hostname, overlap
from factcheck.types import Evidence, ExtractedClaim, MainstreamResult, OutletCheck

OUTLETS = (
    ("Snopes", "snopes.com"),
    ("PolitiFact", "politifact.com"),
    ("FactCheck.org", "factcheck.org"),
    ("AP Fact Check", "apnews.com"),
    ("Reuters Fact Check", "reuters.com"),
)

_AGREE = (
    "mostly true", "mostly accurate", "mostly correct", "true", "accurate", "correct", "confirmed",
)
_DISPUTE = (
    "pants on fire", "four pinocchios", "three pinocchios", "mostly false", "misleading",
    "miscaptioned", "inaccurate", "false", "fake", "incorrect", "distorts", "exaggerates",
)
_MIXED = (
    "half true", "half-true", "mixture", "mixed", "unproven", "needs context", "partly true",
    "partly false", "two pinocchios", "missing context", "labeled satire", "satire",
)
_RATING = re.compile(
    r"\b(pants on fire|four pinocchios|three pinocchios|two pinocchios|mostly false|mostly true|"
    r"mostly accurate|half[- ]true|mixture|mixed|misleading|miscaptioned|unproven|needs context|"
    r"missing context|partly true|partly false|labeled satire|false|true|inaccurate|accurate|"
    r"incorrect|correct|exaggerates|distorts)\b",
    re.I,
)


def polarity(rating: str) -> str:
    text = (rating or "").lower()
    for phrase in _MIXED:
        if phrase in text:
            return "mixed"
    for phrase in _DISPUTE:
        if phrase in text:
            return "disputes"
    for phrase in _AGREE:
        if phrase in text:
            return "agrees"
    return "unclear"


def _rating_from(text: str) -> str:
    match = _RATING.search(text or "")
    return match.group(0) if match else ""


def _outlet_for(url: str, fallback: str = "") -> str:
    host = hostname(url)
    for name, domain in OUTLETS:
        if host == domain or host.endswith("." + domain):
            return name
    return fallback or host or "Fact-checker"


def consensus(checks: list[OutletCheck]) -> str:
    if not checks:
        return "no_coverage"
    votes = [item.polarity for item in checks if item.polarity != "unclear"]
    if not votes:
        return "mixed"
    if all(vote == "agrees" for vote in votes):
        return "mostly_agrees"
    if all(vote == "disputes" for vote in votes):
        return "disputed"
    return "mixed"


def _dedupe(checks: list[OutletCheck]) -> list[OutletCheck]:
    best: dict[str, OutletCheck] = {}
    for item in checks:
        current = best.get(item.outlet)
        if current is None or (item.published_at or "") > (current.published_at or ""):
            best[item.outlet] = item
    return list(best.values())


def coverage_strength(label: str, clear_votes: int) -> float:
    if label == "no_coverage":
        return 0.0
    if label == "mixed":
        return 0.4
    if clear_votes >= 3:
        return 0.8
    if clear_votes == 2:
        return 0.65
    return 0.5


def build_result(checks: list[OutletCheck]) -> MainstreamResult:
    chosen = _dedupe(checks)
    label = consensus(chosen)
    clear = sum(1 for item in chosen if item.polarity != "unclear")
    if not chosen:
        detail = "No matching write-ups were found at the outlets checked."
    else:
        detail = "; ".join(f"{item.outlet}: {item.rating}" for item in chosen)
    reasoning = (
        "What mainstream fact-checkers say: "
        f"{label.replace('_', ' ')}. {detail} "
        "This reports their ratings. It is not a true or false verdict and it is not proof."
    )
    return MainstreamResult(label, chosen, reasoning, coverage_strength(label, clear))


def from_google_payload(payload: dict, claim: str) -> list[OutletCheck]:
    found: list[OutletCheck] = []
    for item in payload.get("claims") or []:
        claim_text = item.get("text") or ""
        if claim_text and overlap(claim, claim_text) < 0.25:
            continue
        for review in item.get("claimReview") or []:
            publisher = review.get("publisher") or {}
            rating = review.get("textualRating") or ""
            url = review.get("url") or ""
            if not url or not rating:
                continue
            if overlap(claim, f"{claim_text} {rating} {review.get('title') or ''}") < 0.2:
                continue
            found.append(
                OutletCheck(
                    outlet=_outlet_for(url, publisher.get("name") or ""),
                    rating=rating,
                    url=url,
                    published_at=review.get("reviewDate"),
                    polarity=polarity(rating),
                )
            )
    return found


def from_search_hits(hits: list[Evidence], claim: str) -> list[OutletCheck]:
    found = []
    for hit in hits:
        if overlap(claim, f"{hit.title} {hit.snippet}") < 0.25:
            continue
        rating = _rating_from(f"{hit.title} {hit.snippet}")
        if not rating:
            continue
        found.append(
            OutletCheck(
                outlet=_outlet_for(hit.url),
                rating=rating,
                url=hit.url,
                published_at=hit.published_at,
                polarity=polarity(rating),
            )
        )
    return found


def site_query(claims: list[ExtractedClaim]) -> str:
    text = " ".join(claim.text for claim in claims)[:240]
    sites = " OR ".join(f"site:{domain}" for _, domain in OUTLETS)
    return f"{text} fact check ({sites})"


class MainstreamChecker:
    def __init__(self, google_search=None, web_search=None):
        self.google_search = google_search
        self.web_search = web_search
        self.queries: list[str] = []

    def check(self, claims: list[ExtractedClaim]) -> list[MainstreamResult]:
        results: list[MainstreamResult] = []
        missed: list[ExtractedClaim] = []
        missed_index: list[int] = []
        for index, claim in enumerate(claims):
            google_hits: list[OutletCheck] = []
            if self.google_search:
                google_hits = list(self.google_search(claim.text) or [])
            if google_hits:
                results.append(build_result(google_hits))
            else:
                results.append(build_result([]))
                missed.append(claim)
                missed_index.append(index)
        if missed and self.web_search:
            query = site_query(missed)
            self.queries.append(query)
            hits = list(self.web_search(query) or [])
            for claim, index in zip(missed, missed_index, strict=True):
                parsed = from_search_hits(hits, claim.text)
                if parsed:
                    results[index] = build_result(parsed)
        return results
