"""Turn a pasted AI answer into short claims and a short search query."""

from __future__ import annotations

import re

from factcheck.types import ExtractedClaim

_SOURCE_LINE = re.compile(r"^(sources?|references?|citations?)\s*[:\-].*$", re.I)
_SOURCE_TAG = re.compile(r"\b(?:source|sources|via|citation)\s*:\s*[^\n.|]{1,60}", re.I)
_OUTLETS = re.compile(
    r"\b(PMC\s*[-–]\s*NIH|PubMed|GoodRx|iHerb|LifeSource Vitamins|WebMD|Healthline|Examine\.com)\b",
    re.I,
)
_OFFER = re.compile(r"\bwant to (compare|see|know|learn)\b", re.I)
_YES = re.compile(r"^(yes|no|sure|ok|okay)[.!]?$", re.I)
_HEADER_CELL = {
    "supplement",
    "supplements",
    "dose",
    "dosage",
    "source",
    "sources",
    "average reduction",
    "reduction",
    "effect",
    "outcome",
    "study",
    "notes",
}


def clean_claim_text(text: str) -> str:
    cleaned = _SOURCE_TAG.sub(" ", text or "")
    cleaned = _OUTLETS.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t-|,;:")
    return cleaned


def _table_row(line: str) -> str | None:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    if not cells:
        return None
    lowered = [cell.lower() for cell in cells]
    if all(cell in _HEADER_CELL or len(cell) < 3 for cell in lowered):
        return None
    kept = []
    for cell in cells:
        if cell.lower() in _HEADER_CELL or _OUTLETS.search(cell) or _SOURCE_LINE.match(cell):
            continue
        piece = clean_claim_text(cell)
        if piece:
            kept.append(piece)
    if len(kept) < 2:
        return None
    text = " ".join(kept)
    return text if text.endswith(".") else text + "."


def prepare_answer(text: str) -> str:
    """Drop citation chrome, offers, and questions. Split table rows into their own lines."""
    kept: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.count("|") >= 2:
            row = _table_row(line)
            if row:
                kept.append(row)
            continue
        if _SOURCE_LINE.match(line) or _YES.match(line) or _OFFER.search(line) or line.endswith("?"):
            continue
        cleaned = clean_claim_text(line)
        if _OFFER.search(cleaned) or cleaned.endswith("?") or len(cleaned.split()) < 4:
            continue
        kept.append(cleaned if cleaned.endswith((".", "!", '"')) else cleaned)
    return "\n".join(kept)


def focus_query(text: str, limit_words: int = 16) -> str:
    """A short retrieval query. Source labels and table chrome are not part of it."""
    cleaned = clean_claim_text(prepare_answer(text) or text)
    words = [word for word in cleaned.replace("|", " ").split() if word]
    return " ".join(words[:limit_words])[:160]


def priority(claim: ExtractedClaim) -> tuple[int, int, int]:
    """Lower sorts first. Dosage, medical, and numeric claims are checked before the rest."""
    category_rank = {"T4": 0, "T1": 1, "T2": 2, "T3": 3, "T5": 4, "T6": 5, "T7": 6, "T8": 8}.get(
        claim.claim_category, 5
    )
    stakes = 0 if claim.fragile or claim.claim_category in {"T1", "T4"} else 1
    kind = 0 if claim.claim_type == "fact" else 1
    return (stakes, category_rank, kind)
