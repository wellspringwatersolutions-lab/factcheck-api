"""Passage-level entailment. A label counts only when its quote is in the passage."""

from __future__ import annotations

from factcheck.checks import entailment, quote_appears, quantity_conflict, split_passages
from factcheck.llm import loads_object
from factcheck.textutil import overlap
from factcheck.types import Evidence, ExtractedClaim, JudgeVote, UsageMeter

_PROMPT = """STAGE: entailment
You label each passage against the claim. You do not decide the final verdict.
For every passage return SUPPORTS, REFUTES, or NOT_ENOUGH_INFO.
quote must be copied verbatim from that passage. If you cannot quote it, use NOT_ENOUGH_INFO and an empty quote.
Ignore HTML, scripts, and boilerplate. A negation in an unrelated sentence is not a refutation.
Return JSON {"passages": [{"url": "...", "label": "SUPPORTS|REFUTES|NOT_ENOUGH_INFO", "quote": "verbatim words"}]}"""

_LABELS = {
    "SUPPORTS": "supports",
    "REFUTES": "contradicts",
    "NOT_ENOUGH_INFO": "not_relevant",
    "supports": "supports",
    "contradicts": "contradicts",
    "not_relevant": "not_relevant",
}


def filter_with_votes(passages: list[Evidence], votes: list[JudgeVote]) -> list[Evidence]:
    """Drop a passage only when every judge calls it irrelevant and it does not entail or contradict."""
    if not votes:
        return passages
    flagged: set[str] = set()
    for vote in votes:
        flagged.update(url for url, stance in vote.stances.items() if stance == "not_relevant")
    if len(votes) >= 2:
        agreed = set(votes[0].stances)
        for vote in votes[1:]:
            agreed &= {url for url, stance in vote.stances.items() if stance == "not_relevant"}
        flagged = agreed
    for passage in passages:
        if passage.url in flagged and passage.entailment == "neutral":
            passage.used = False
    return passages


def offline_label(claim: str, page: str) -> tuple[str, str]:
    """Sentence-level fallback used when no model is called. Negation stays inside one sentence."""
    chunks = split_passages(page) or [page.strip()]
    if not chunks or not chunks[0]:
        return "not_relevant", ""
    best = max(chunks, key=lambda chunk: overlap(claim, chunk))
    conflict = quantity_conflict(claim, best)
    if conflict:
        return "contradicts", conflict
    relation = entailment(claim, best)
    if relation == "entails":
        return "supports", best[:500]
    if relation == "contradicts":
        return "contradicts", best[:500]
    return "not_relevant", ""


def apply_verified_labels(passages: list[Evidence], votes: list[JudgeVote]) -> list[Evidence]:
    """Keep a SUPPORTS or REFUTES label only when its quote is in the cleaned passage."""
    for passage in passages:
        if passage.url_status is not None and passage.url_status >= 400:
            passage.entailment = "neutral"
            passage.used = False
            passage.quote = ""
            continue
        if passage.quote_match is False:
            passage.entailment = "neutral"
            passage.used = False
            passage.quote = ""
            continue
        text = passage.page_text or passage.snippet or ""
        verified: list[str] = []
        chosen_quote = ""
        for vote in votes:
            label = vote.stances.get(passage.url)
            quote = (vote.quotes or {}).get(passage.url, "")
            if label not in {"supports", "contradicts"}:
                continue
            if quote and quote_appears(text, quote):
                verified.append(label)
                if not chosen_quote:
                    chosen_quote = quote
        if passage.entailment == "contradicts" and passage.used and passage.quote:
            # A numeric or date mismatch already recorded on this passage stays.
            continue
        if not verified:
            passage.entailment = "neutral"
            passage.used = False
            passage.quote = ""
            continue
        passage.quote = chosen_quote
        if "supports" in verified and "contradicts" in verified:
            passage.entailment = "conflicts"
        elif "contradicts" in verified:
            passage.entailment = "contradicts"
        else:
            passage.entailment = "entails"
        passage.used = True
    return passages


class HeuristicRanker:
    def __init__(self, role: str, model: str, meter: UsageMeter | None = None):
        self.role = role
        self.model = model
        self.meter = meter
        self.family = model.split("/")[0]

    def rank(self, claims: list[ExtractedClaim], passages: list[Evidence]) -> list[JudgeVote]:
        if self.meter:
            self.meter.add_tokens(
                model=self.model,
                provider="heuristic",
                input_tokens=120,
                output_tokens=40,
                cost=0.00002,
            )
        votes = []
        for claim in claims:
            stances = {}
            quotes = {}
            for item in passages:
                label, quote = offline_label(claim.text, item.page_text or item.snippet)
                stances[item.url] = label
                if quote:
                    quotes[item.url] = quote
            votes.append(
                JudgeVote(
                    role=self.role,
                    model=self.model,
                    provider="heuristic",
                    family=self.family,
                    verdict="rank_only",
                    reasoning="Labeled each passage from its own text. This is not a verdict.",
                    stances=stances,
                    ranked_urls=[url for url, label in stances.items() if label != "not_relevant"],
                    quotes=quotes,
                )
            )
        return votes


class LiveRanker:
    def __init__(self, client, role: str, model: str, meter: UsageMeter, settings):
        self.client = client
        self.role = role
        self.model = model
        self.meter = meter
        self.settings = settings

    def rank(self, claims: list[ExtractedClaim], passages: list[Evidence]) -> list[JudgeVote]:
        if not claims or not passages:
            return []
        lines = []
        for index, claim in enumerate(claims):
            lines.append(f"CLAIM {index}: {claim.text}")
        lines.append("PASSAGES:")
        for item in passages:
            chunks = split_passages(item.page_text or item.snippet) or [(item.page_text or item.snippet)[:700]]
            shown = "\n".join(f"  - {chunk}" for chunk in chunks[:4])
            lines.append(f"- url: {item.url}\ntitle: {item.title}\npassages:\n{shown}")
        from factcheck.llm import estimate_cost

        result = self.client.complete(
            model=self.model,
            messages=[{"role": "system", "content": _PROMPT}, {"role": "user", "content": "\n".join(lines)}],
            temperature=0.0,
        )
        self.meter.add_tokens(
            model=result.model,
            provider=result.provider,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost=estimate_cost(self.settings, result.model, result.input_tokens, result.output_tokens),
        )
        try:
            payload = loads_object(result)
        except ValueError:
            return [
                JudgeVote(self.role, result.model, result.provider, result.family, "rank_only", "unparsed labels", {}, [], {})
                for _ in claims
            ]
        by_url = {}
        for item in payload.get("passages") or []:
            url = item.get("url")
            if isinstance(url, str):
                by_url[url] = item
        votes = []
        for _claim in claims:
            stances = {}
            quotes = {}
            for passage in passages:
                item = by_url.get(passage.url) or {}
                label = _LABELS.get(str(item.get("label") or ""), "not_relevant")
                quote = str(item.get("quote") or "")
                stances[passage.url] = label
                if quote:
                    quotes[passage.url] = quote
            votes.append(
                JudgeVote(
                    role=self.role,
                    model=result.model,
                    provider=result.provider,
                    family=result.family,
                    verdict="rank_only",
                    reasoning="Labeled passages. A quote that is not in the passage is ignored.",
                    stances=stances,
                    ranked_urls=[url for url, label in stances.items() if label != "not_relevant"],
                    quotes=quotes,
                )
            )
        return votes
