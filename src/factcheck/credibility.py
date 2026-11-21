"""Domain grade, source type, freshness, and syndicated-copy dedupe."""

from __future__ import annotations

import hashlib
import re
from datetime import date

from factcheck.textutil import hostname, jaccard, normalize_text, overlap, parse_date
from factcheck.types import Evidence

GRADE_WEIGHT = {"A": 1.0, "B": 0.85, "C": 0.6, "D": 0.35, "E": 0.2, "F": 0.1}

# Default weights. A weak paper can still lose to a stronger record of another type.
TYPE_WEIGHT = {
    "systematic_review": 1.0,
    "rct": 0.85,
    "textbook": 0.9,
    "primary": 0.9,
    "data": 0.95,
    "observational": 0.55,
    "secondary": 0.45,
    "expert_opinion": 0.35,
    "eyewitness": 0.2,
    "anecdote": 0.1,
    "syndicated": 0.15,
}

T4_WEIGHT = {
    "systematic_review": 1.0,
    "rct": 0.85,
    "observational": 0.5,
    "expert_opinion": 0.25,
    "primary": 0.8,
    "data": 0.7,
    "secondary": 0.3,
}

_A_HOSTS = {
    "who.int", "cdc.gov", "nih.gov", "nasa.gov", "nature.com", "science.org",
    "nejm.org", "thelancet.com", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
}
_B_SUFFIXES = (".edu", ".ac.uk")
_B_HOSTS = {
    "nytimes.com", "washingtonpost.com", "theguardian.com", "ft.com", "wsj.com",
    "economist.com", "springer.com", "wiley.com", "ieee.org", "acm.org",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "semanticscholar.org", "arxiv.org",
    "snopes.com", "politifact.com", "factcheck.org", "britannica.com",
}
_D_HOSTS = {"medium.com", "substack.com", "blogspot.com", "wordpress.com", "tumblr.com"}
_F_HOSTS = {"reddit.com", "facebook.com", "x.com", "twitter.com", "tiktok.com", "instagram.com", "quora.com", "4chan.org"}


def grade_for(url: str) -> str:
    if url.startswith(("scifact://",)):
        return "A"
    if url.startswith(("fever://", "heldout://", "corpus://")):
        return "B"
    host = hostname(url)
    if not host:
        return "E"
    if host.endswith(".gov") or host.endswith(".gov.uk") or host in _A_HOSTS:
        return "A"
    if host in _B_HOSTS or host.endswith(_B_SUFFIXES):
        return "B"
    if host in _F_HOSTS:
        return "F"
    if host in _D_HOSTS:
        return "D"
    return "E"


def infer_source_type(url: str, title: str, snippet: str) -> str:
    host = hostname(url)
    blob = f"{title} {snippet}".lower()
    if "systematic review" in blob or "meta-analysis" in blob or "cochrane" in blob:
        return "systematic_review"
    if re.search(r"\brandomi[sz]ed\b|\brct\b", blob):
        return "rct"
    if any(token in blob for token in ("cohort", "case-control", "observational", "cross-sectional")):
        return "observational"
    if any(token in blob for token in ("editorial", "commentary", "expert opinion")):
        return "expert_opinion"
    if "i saw" in blob or "eyewitness" in blob:
        return "eyewitness"
    if host in {"britannica.com"} or "textbook" in blob:
        return "textbook"
    if host.endswith("data.gov") or host in {"census.gov", "ourworldindata.org"}:
        return "data"
    if any(token in host for token in ("pubmed", "ncbi.nlm.nih.gov", "nature.com", "nejm.org", "thelancet.com", "science.org", "cell.com")):
        return "primary"
    if host.endswith(".gov") or host.endswith(".gov.uk") or url.startswith("scifact://"):
        return "primary"
    return "secondary"


def source_weight(evidence: Evidence, category: str) -> float:
    grade = GRADE_WEIGHT.get(evidence.grade, 0.2)
    if evidence.source_type == "peer_reviewed":
        type_weight = {"systematic_review": 1.0, "rct": 0.85, "observational": 0.55, "review": 0.7}.get(
            evidence.study_type, 0.6
        )
        return grade * type_weight * evidence.freshness
    if category == "T4":
        type_weight = T4_WEIGHT.get(evidence.source_type, TYPE_WEIGHT.get(evidence.source_type, 0.4))
    elif category == "T5" and evidence.source_type in {"primary", "data"}:
        type_weight = 1.0
    elif category == "T5" and evidence.source_type == "secondary":
        type_weight = 0.4
    elif category in {"T2", "T3"} and evidence.grade in {"A", "B"}:
        type_weight = max(TYPE_WEIGHT.get(evidence.source_type, 0.4), 0.9)
    elif category == "T7":
        type_weight = min(TYPE_WEIGHT.get(evidence.source_type, 0.4), 0.25)
    elif category == "T6" and evidence.source_type == "expert_opinion":
        type_weight = 0.3
    else:
        type_weight = TYPE_WEIGHT.get(evidence.source_type, 0.4)
    return grade * type_weight * evidence.freshness


def annotate(evidence: Evidence, *, time_sensitive: bool, today: date | None = None, stale_after_days: int = 120) -> Evidence:
    today = today or date.today()
    scholarly = evidence.source_type == "peer_reviewed" or evidence.provider in {
        "openalex",
        "semantic_scholar",
        "pubmed",
        "europepmc",
    }
    preserved = evidence.source_type in {"fact_checker", "structured", "primary_source"} or evidence.provider in {
        "google_factcheck",
        "wikidata",
        "loc",
    }
    if preserved:
        evidence.grade = evidence.grade or "B"
    elif scholarly:
        if not evidence.grade or evidence.grade == "E":
            evidence.grade = {"systematic_review": "A", "rct": "B"}.get(evidence.study_type, "B")
        evidence.source_type = "peer_reviewed"
    elif evidence.url.startswith("http"):
        evidence.grade = grade_for(evidence.url)
        evidence.source_type = infer_source_type(evidence.url, evidence.title, evidence.snippet)
    elif not evidence.grade:
        evidence.grade = grade_for(evidence.url)
    published = parse_date(evidence.published_at)
    evidence.stale = False
    evidence.freshness = 1.0
    if time_sensitive:
        if published is None:
            evidence.freshness = 0.75
        else:
            age = (today - published).days
            if age > stale_after_days:
                evidence.stale = True
                evidence.freshness = 0.5
            elif age > stale_after_days // 2:
                evidence.freshness = 0.8
    if not evidence.origin_id:
        evidence.origin_id = hashlib.sha256(evidence.url.encode()).hexdigest()[:16]
    return evidence


def _title_key(title: str) -> str:
    text = title.lower()
    text = re.sub(r"\b(reuters|ap news|associated press|bbc|cnn|via)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def same_story(left: Evidence, right: Evidence) -> bool:
    if left.url == right.url:
        return True
    title_left, title_right = _title_key(left.title), _title_key(right.title)
    if title_left and title_right and (title_left == title_right or jaccard(title_left, title_right) >= 0.72):
        return True
    if overlap(left.snippet, right.snippet) >= 0.8 and jaccard(left.snippet, right.snippet) >= 0.55:
        return True
    return False


def dedupe_syndicated(sources: list[Evidence]) -> list[Evidence]:
    """Keep the strongest origin of a story. Repeats are not independent sources."""
    kept: list[Evidence] = []
    for source in sources:
        match = next((item for item in kept if same_story(item, source)), None)
        if match is None:
            kept.append(source)
            continue
        prefer_new = (GRADE_WEIGHT.get(source.grade, 0), TYPE_WEIGHT.get(source.source_type, 0)) > (
            GRADE_WEIGHT.get(match.grade, 0),
            TYPE_WEIGHT.get(match.source_type, 0),
        )
        winner, loser = (source, match) if prefer_new else (match, source)
        winner.syndicated_also = list(dict.fromkeys([*winner.syndicated_also, loser.url]))
        winner.origin_id = winner.origin_id or match.origin_id
        if prefer_new:
            kept[kept.index(match)] = winner
    for item in kept:
        if not item.origin_id:
            item.origin_id = hashlib.sha256(normalize_text(item.title or item.url).encode()).hexdigest()[:16]
    return kept


def dedupe_urls(sources: list[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    kept = []
    for source in sources:
        key = source.url.split("#")[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        kept.append(source)
    return kept
