"""Claim categories T1–T8. The ranking is a default, not a verdict by itself."""

from __future__ import annotations

import re

from factcheck.textutil import has_number, is_fragile, is_time_sensitive
from factcheck.types import ExtractedClaim

CATEGORIES: dict[str, str] = {
    "T1": "Math/logic",
    "T2": "Established scientific law",
    "T3": "Strongly supported theory",
    "T4": "Empirical or medical result",
    "T5": "Historical fact",
    "T6": "Expert consensus or testimony",
    "T7": "Eyewitness or personal account",
    "T8": "Anecdote, speculation, or opinion",
}

_OPINION = re.compile(
    r"\b(i think|i believe|in my opinion|arguably|ridiculous|greatest|worst|"
    r"everyone knows|obviously|should|shouldn't|ought|probably|perhaps|"
    r"is a disaster|is terrible|is wonderful|best ever|narrative)\b",
    re.I,
)
_EYEWITNESS = re.compile(r"\b(i saw|i watched|eyewitness|told me|my experience|i heard)\b", re.I)
_THEORY = re.compile(
    r"\b(germ theory|relativity|evolution|natural selection|plate tectonics|"
    r"heliocentrism|quantum mechanics|atomic theory)\b",
    re.I,
)
_LAW = re.compile(
    r"\b(law of|speed of light|boils at|conservation of|newton's|thermodynamics|"
    r"photosynthesis|periodic table|gravity)\b",
    re.I,
)
_MEDICAL = re.compile(
    r"\b(trial|randomized|randomised|meta-analysis|patients|vaccine|dosage|"
    r"clinical|cohort|placebo|risk ratio|treatment)\b",
    re.I,
)
_EXPERT = re.compile(
    r"\b(experts say|scientists believe|consensus|analysts|according to researchers)\b",
    re.I,
)
_MATH = re.compile(
    r"(\d+(?:\.\d+)?\s*[\+\-\*/\^x×]\s*\d+(?:\.\d+)?|"
    r"\b(equals|sqrt|sum of|product of)\b)",
    re.I,
)
_HYPOTHESIS = re.compile(
    r"\b(we predict|novel conjecture|business idea|startup idea|going to|"
    r"might someday|in the future|if we (?:build|launch|ship)|hypothetically)\b",
    re.I,
)
_FUTURE_YEAR = re.compile(r"\b(20[3-9]\d|2[1-9]\d{2})\b")
_WILL = re.compile(r"\b(will|won't|shall)\b", re.I)
_PAST_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2]\d)\b")


def is_math(text: str) -> bool:
    return bool(_MATH.search(text or ""))


def is_hypothesis(text: str) -> bool:
    """Future predictions, novel conjectures, and business ideas. Not value judgments."""
    blob = text or ""
    if _HYPOTHESIS.search(blob):
        return True
    if _WILL.search(blob) and (_FUTURE_YEAR.search(blob) or not _PAST_YEAR.search(blob)):
        if not _OPINION.search(blob):
            return True
    return False


def classify_category(text: str, claim_type: str = "fact") -> str:
    blob = text or ""
    if claim_type == "opinion" or _OPINION.search(blob):
        return "T8"
    if is_math(blob):
        return "T1"
    if _EYEWITNESS.search(blob):
        return "T7"
    if _THEORY.search(blob):
        return "T3"
    if _LAW.search(blob):
        return "T2"
    if _MEDICAL.search(blob):
        return "T4"
    if _EXPERT.search(blob):
        return "T6"
    if _PAST_YEAR.search(blob) and not is_time_sensitive(blob):
        return "T5"
    if is_hypothesis(blob):
        return "T8"
    return "T6"


def normalize_claim(claim: ExtractedClaim) -> ExtractedClaim:
    if claim.claim_type == "opinion":
        claim.claim_category = "T8"
    elif not claim.claim_category:
        claim.claim_category = classify_category(claim.text, claim.claim_type)
    if claim.claim_category == "T8":
        claim.claim_type = "opinion"
    if claim.claim_category == "T1":
        claim.claim_type = "fact"
        claim.hypothesis = False
    if claim.claim_category not in CATEGORIES:
        claim.claim_category = classify_category(claim.text, claim.claim_type)
    if claim.claim_category == "T1":
        claim.hypothesis = False
    elif claim.claim_category in {"T2", "T3", "T4", "T5", "T6", "T7"}:
        claim.hypothesis = False
    elif claim.hypothesis or is_hypothesis(claim.text):
        claim.hypothesis = True
    claim.fragile = is_fragile(claim.text) or claim.time_sensitive or has_number(claim.text)
    claim.time_sensitive = claim.time_sensitive or is_time_sensitive(claim.text)
    claim.scientific = claim.scientific or claim.claim_category in {"T2", "T3", "T4"}
    return claim


def effort_for(claim: ExtractedClaim) -> str:
    """Where to spend retrieval. Fragile details get a stricter pass."""
    if claim.claim_category == "T1":
        return "compute"
    if claim.hypothesis:
        return "hypothesis"
    if claim.claim_category == "T8" or claim.claim_type == "opinion":
        return "opinion"
    if claim.claim_category in {"T2", "T3"} and not claim.fragile:
        return "light"
    if claim.fragile or claim.claim_category in {"T4", "T5", "T6", "T7"}:
        return "strict"
    return "standard"


EFFORT_LIMITS = {
    "light": {"search": 3, "crawl": 0},
    "standard": {"search": 5, "crawl": 1},
    "strict": {"search": 8, "crawl": 3},
    "hypothesis": {"search": 5, "crawl": 1},
}
