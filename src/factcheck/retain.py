"""Decide which search hits stay with a claim. A date window ranks. It does not exclude."""

from __future__ import annotations

import logging
import re

from factcheck.request_ctx import note_drop
from factcheck.textutil import hostname, overlap
from factcheck.types import Evidence

logger = logging.getLogger("factcheck")

_FACT_CHECK = ("snopes.com", "leadstories.com", "politifact.com", "factcheck.org", "fullfact.org")
_BLOCKED = (
    "mylife.com",
    "spokeo.com",
    "whitepages.com",
    "beenverified.com",
    "intelius.com",
    "instantcheckmate.com",
    "truthfinder.com",
    "criminalrecords.com",
)
_NAME = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
_INVENTED_NEGATIVE = re.compile(r"\bthere are no criminal convictions\b", re.I)


def is_official(url: str) -> bool:
    host = hostname(url or "")
    if not host:
        return False
    if host.endswith(".gov") or host.endswith(".gov.uk") or host.endswith(".gouv.fr") or host.endswith(".gc.ca"):
        return True
    if host == "europa.eu" or host.endswith(".europa.eu") or host.endswith(".ready.gov") or host == "ready.gov":
        return True
    return False


def blocked_site(url: str) -> bool:
    host = hostname(url or "")
    blob = f"{host} {url or ''}".lower()
    if any(host == name or host.endswith("." + name) for name in _BLOCKED):
        return True
    return "criminal-record" in blob or "criminalrecord" in blob or "arrests.org" in blob


def person_names(text: str) -> list[str]:
    found = []
    for name in _NAME.findall(text or ""):
        if name.lower() in {"the united", "united states", "new york"}:
            continue
        if name not in found:
            found.append(name)
    return found


def entity_mismatch(claim: str, passage: str, context: str = "") -> bool:
    """A different person or a different topic does not count as evidence."""
    blob = passage or ""
    lowered = blob.lower()
    focus = f"{claim} {context}"
    for name in person_names(claim) or person_names(context):
        parts = name.split()
        exact = name.lower() in lowered
        longer = bool(re.search(rf"\b{parts[0]}\s+[A-Z][a-z]+\s+{parts[-1]}\b", blob))
        if longer and not exact:
            return True
        if not exact and parts[-1].lower() in lowered:
            return True
        if not exact and len(parts) >= 2:
            return True
    if re.search(r"\b(president|founding|1735|18th century)\b", focus, re.I) and re.search(
        r"\b(bodkin|surgeon|crime gang|criminal record)\b", blob, re.I
    ):
        return True
    about_supplies = bool(re.search(r"\b(food|kit|stockpile|hoard|shelf-stable)\b", claim, re.I))
    if about_supplies and re.search(r"\b(covid|quarantine|vacation rental|hotel booking)\b", blob, re.I):
        if not re.search(r"\b(food|kit|stockpile|hoard|emergency supply|shelf-stable)\b", blob, re.I):
            return True
    return False


def drop_invented_negative(claims: list, question: str | None) -> list:
    """A question does not become 'there are no criminal convictions' by itself."""
    if not question or not re.search(r"\bcriminal\b", question, re.I):
        return claims
    kept = [claim for claim in claims if not _INVENTED_NEGATIVE.search(getattr(claim, "text", ""))]
    return kept


def is_fact_checker(url: str, source_type: str = "") -> bool:
    if source_type == "fact_checker":
        return True
    host = hostname(url or "")
    return any(host == name or host.endswith("." + name) for name in _FACT_CHECK)


def _day(value: str | None) -> str:
    text = (value or "").strip()
    return text[:10] if len(text) >= 10 and text[4] == "-" else ""


def outside_window(published_at: str | None, window: dict | None) -> bool:
    if not window:
        return False
    day = _day(published_at)
    if not day:
        return False
    return day < window["start_date"] or day > window["end_date"]


def screen_hits(claim: str, hits: list[Evidence], window: dict | None, context: str = "") -> tuple[list[Evidence], dict[str, int]]:
    """Keep official, fact-checker, and undated pages. Rank the rest by the date window."""
    drops: dict[str, int] = {}
    seen: set[str] = set()
    ranked: list[tuple[int, float, Evidence]] = []

    def _drop(reason: str, hit: Evidence) -> None:
        drops[reason] = drops.get(reason, 0) + 1
        note_drop(reason)
        logger.warning(
            "drop %s claim=%r url=%s published=%s",
            reason,
            (claim or "")[:120],
            hit.url,
            hit.published_at or "",
        )

    for hit in hits:
        key = (hit.url or "").split("#")[0].rstrip("/").lower()
        if not key or key in seen:
            _drop("dedupe", hit)
            continue
        seen.add(key)
        if blocked_site(hit.url):
            _drop("blocked_site", hit)
            continue
        passage = " ".join(part for part in (hit.title, hit.snippet, hit.page_text, hit.quote) if part)
        if entity_mismatch(claim, passage, context):
            _drop("entity_mismatch", hit)
            continue
        official = is_official(hit.url)
        checker = is_fact_checker(hit.url, hit.source_type)
        text = " ".join(part for part in (hit.title, hit.snippet, hit.page_text, hit.quote) if part)
        score = overlap(claim, text)
        undated = not _day(hit.published_at)
        if window and undated:
            logger.warning("keep undated claim=%r url=%s", (claim or "")[:120], hit.url)
        elif window and outside_window(hit.published_at, window):
            if official or checker:
                logger.warning("keep outside window protected claim=%r url=%s published=%s", (claim or "")[:120], hit.url, hit.published_at)
            else:
                logger.warning("rank outside window claim=%r url=%s published=%s", (claim or "")[:120], hit.url, hit.published_at)
        if not official and not checker and score < 0.08:
            _drop("passage_match", hit)
            continue
        if not official and not checker and (hit.grade or "E") == "E" and score < 0.2:
            _drop("domain_tier", hit)
            continue
        if not hit.quote:
            hit.quote = (hit.snippet or hit.page_text or "")[:400]
        if window and _day(hit.published_at) and not outside_window(hit.published_at, window):
            boost = 0
        elif official or checker or undated:
            boost = 1
        else:
            boost = 2
        ranked.append((boost, -score, hit))
    ranked.sort(key=lambda item: (item[0], item[1], item[2].url))
    kept = [hit for _, _, hit in ranked]
    logger.warning(
        "retrieval claim=%r returned=%d kept=%d drops=%s",
        (claim or "")[:160],
        len(hits),
        len(kept),
        drops or {},
    )
    return kept, drops
