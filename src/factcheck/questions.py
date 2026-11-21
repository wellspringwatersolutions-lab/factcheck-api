"""Draft a short answer to a bare question, then let the normal checker judge it.

The draft is never a source. Verdicts still require a retrieved passage or a tool result.
"""

from __future__ import annotations

from factcheck.llm import estimate_cost
from factcheck.textutil import parse_json_object

_DRAFT = """Answer the question in two to four factual sentences.
Put the direct answer first. Add one caveat only when it belongs in the usual account.
Do not include URLs or a source list.
stance is the direct answer: yes, no, partly, or not_enough_evidence.
Use not_enough_evidence only when you cannot answer.
Return JSON {"answer": "...", "stance": "yes|no|partly|not_enough_evidence"}"""

_STANCES = {"yes", "no", "partly", "not_enough_evidence"}


def stance_of_text(text: str) -> str:
    """Read a direct answer from the first words when the model omitted stance."""
    lowered = (text or "").strip().lower()
    if lowered.startswith("unconfirmed"):
        return "not_enough_evidence"
    if lowered.startswith("correction:"):
        lowered = lowered.split("\n", 1)[-1].strip()
    if re_match_no(lowered):
        return "no"
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("partly") or lowered.startswith("in part"):
        return "partly"
    return ""


def re_match_no(lowered: str) -> bool:
    return lowered.startswith("no") and (len(lowered) == 2 or not lowered[2].isalpha())


def parse_stance(raw: str, text: str) -> str:
    cleaned = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in {"not_enough", "unconfirmed", "not_enough_evidence"}:
        return "not_enough_evidence"
    if cleaned in _STANCES:
        return cleaned
    return stance_of_text(text) or "not_enough_evidence"


class DraftError(RuntimeError):
    """The draft call failed. Nothing was checked and the caller is not charged."""

    def __init__(self, message: str = "The draft answer could not be written, so nothing was checked."):
        super().__init__(message)
        self.message = message

    def body(self) -> dict:
        return {"error": {"code": "draft_failed", "message": self.message}}


class AnswerDrafter:
    """One short Chutes (or failover) call. Timeout is 8 seconds."""

    def __init__(self, client, model: str, meter, settings):
        self.client = client
        self.model = model
        self.meter = meter
        self.settings = settings

    def __call__(self, question: str) -> tuple[str, str, str]:
        result = self.client.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": _DRAFT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=400,
            timeout=8.0,
        )
        if self.meter is not None:
            self.meter.add_tokens(
                model=result.model,
                provider=result.provider,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost=estimate_cost(self.settings, result.model, result.input_tokens, result.output_tokens),
            )
        try:
            payload = parse_json_object(result.content)
        except (ValueError, TypeError) as exc:
            raise DraftError("The draft answer could not be written, so nothing was checked.") from exc
        answer = payload.get("answer") or payload.get("text") or ""
        text = str(answer).strip()
        if not text:
            raise DraftError("The draft answer was empty, so nothing was checked.")
        stance = parse_stance(str(payload.get("stance") or ""), text)
        return result.model, text, stance
