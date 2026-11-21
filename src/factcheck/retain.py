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
_INVENTED_NEGATIVE = re.compile(
    r"\b(there are no criminal convictions|no historical record|no credible allegation|"
    r"no criminal record|there is no record|was not a criminal|never charged|never convicted|no convictions)\b",
    re.I,
)
_COURT = ("courtlistener.com", "vlex.com", "casetext.com", "justia.com")
_TRAVEL = ("tripadvisor.", "booking.com", "expedia.", "kayak.", "lonelyplanet.", "travelocity.")
_CRIME_OTHER = re.compile(r"\b(pleaded guilty|pled guilty|sex trafficking|husband|wife)\b", re.I)
_LEADING = {"the", "a", "an"}


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


def court_record(url: str) -> bool:
    host = hostname(url or "")
    if any(host == name or host.endswith("." + name) for name in _COURT):
        return True
    return "justia.com" in host and "/cases" in (url or "")


def is_travel_page(url: str, text: str = "") -> bool:
    host = hostname(url or "")
    if any(token in host for token in _TRAVEL):
        return True
    return bool(re.search(r"\b(trip planning|vacation rental|things to do in)\b", text or "", re.I))


def is_prepper_blog(url: str, text: str = "", grade: str = "") -> bool:
    if grade in {"A", "B"}:
        return False
    return bool(re.search(r"\b(prepper|shtf|survivalist|doomsday)\b", f"{hostname(url or '')} {text or ''}", re.I))


def disambiguators(text: str) -> set[str]:
    found = set()
    if re.search(r"\bpresident\b", text or "", re.I):
        found.add("president")
    if re.search(r"\bfounding\b", text or "", re.I):
        found.add("founding")
    if re.search(r"\b(18th century|17[0-9]{2}|180[0-1])\b", text or "", re.I):
        found.add("era")
    return found


def person_names(text: str) -> list[str]:
    found = []
    for name in _NAME.findall(text or ""):
        if name.lower() in {"the united", "united states", "new york"}:
            continue
        if name not in found:
            found.append(name)
    return found


def _subject_name(title: str, body: str) -> str:
    named = person_names(title)
    if named:
        return named[0]
    first = (body or "").split(".")[0]
    named = person_names(first)
    return named[0] if named else ""


def _core_name(name: str) -> list[str]:
    return [part for part in name.split() if part.lower() not in _LEADING]


def _different_person(claim_name: str, subject: str) -> bool:
    claim_parts = _core_name(claim_name)
    subject_parts = _core_name(subject)
    if not claim_parts or not subject_parts:
        return False
    if [part.lower() for part in claim_parts] == [part.lower() for part in subject_parts]:
        return False
    if (
        len(subject_parts) == len(claim_parts) + 1
        and subject_parts[0].lower() == claim_parts[0].lower()
        and subject_parts[-1].lower() == claim_parts[-1].lower()
    ):
        return True
    if (
        len(claim_parts) >= 2
        and len(subject_parts) >= 2
        and claim_parts[-1].lower() == subject_parts[-1].lower()
        and claim_parts[0].lower() != subject_parts[0].lower()
    ):
        return True
    return False


def same_entity(claim: str, passage: str, context: str = "", url: str = "", title: str = "") -> bool | None:
    """True, False, or None when a model should be asked. Court sites need a strong match."""
    focus = f"{claim} {context}"
    names = person_names(claim) or person_names(context)
    claim_name = min(names, key=len) if names else ""
    subject = _subject_name(title, passage)
    if _different_person(claim_name, subject):
        return False
    marks = disambiguators(focus)
    shared = marks & disambiguators(passage)
    if court_record(url) and marks and not shared:
        return False
    blob = f"{title} {passage}"
    if marks and _CRIME_OTHER.search(blob) and not shared:
        return False
    if claim_name and claim_name.lower() not in blob.lower():
        if subject or court_record(url):
            return False
        return None
    if claim_name and claim_name.lower() in blob.lower():
        return True
    return None


def entity_mismatch(claim: str, passage: str, context: str = "", url: str = "", title: str = "") -> bool:
    """A different person or a different topic does not count as evidence."""
    verdict = same_entity(claim, passage, context, url, title)
    if verdict is False:
        return True
    about_supplies = bool(re.search(r"\b(food|kit|stockpile|hoard|shelf-stable)\b", claim, re.I))
    blob = passage or ""
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


def screen_hits(
    claim: str,
    hits: list[Evidence],
    window: dict | None,
    context: str = "",
    entity_judge=None,
) -> tuple[list[Evidence], dict[str, int]]:
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
        body = " ".join(part for part in (hit.snippet, hit.page_text, hit.quote) if part)
        passage = " ".join(part for part in (hit.title, body) if part)
        verdict = same_entity(claim, body, context, hit.url, hit.title)
        if verdict is None and entity_judge is not None:
            verdict = bool(entity_judge(claim, passage, hit))
        if verdict is False or (verdict is None and entity_mismatch(claim, passage, context, hit.url, hit.title)):
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
    reputable = [
        hit
        for hit in kept
        if hit.grade in {"A", "B"} or is_official(hit.url) or is_fact_checker(hit.url, hit.source_type)
    ]
    if reputable:
        kept = reputable
    kept = kept[:4]
    logger.warning(
        "retrieval claim=%r returned=%d kept=%d drops=%s",
        (claim or "")[:160],
        len(hits),
        len(kept),
        drops or {},
    )
    return kept, drops
