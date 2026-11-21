"""Public JSON shapes. Verdicts summarize evidence. They do not prove truth."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from factcheck.branding import SERVICE_NAME

Verdict = Literal[
    "Supported",
    "Contradicted",
    "Unverified",
    "Conflicting",
    "Unsupported",
    "Hypothesis",
]
Consensus = Literal["mostly_agrees", "disputed", "mixed", "no_coverage"]
HypothesisStatus = Literal["supporting_evidence", "contradicting_evidence", "untested"]


class SourceOut(BaseModel):
    url: str
    title: str
    snippet: str
    quote: str = ""
    published_at: str | None = None
    grade: str
    stance: str
    source_type: str
    study_type: str = ""
    origin_id: str = ""
    stale: bool = False
    content_sha256: str = ""
    url_status: int | None = None
    quote_match: bool | None = None
    number_match: bool | None = None
    date_match: bool | None = None
    entailment: str = ""
    used: bool = False
    syndicated_also: list[str] = Field(default_factory=list)
    provider: str | None = None


class JudgeOut(BaseModel):
    """A judge only filters and ranks passages. Its text is not a verdict."""

    model: str
    provider: str
    role: str
    ranked_urls: list[str] = Field(default_factory=list)
    filtered_urls: list[str] = Field(default_factory=list)
    note: str = ""


class JudgesOut(BaseModel):
    a: JudgeOut | None = None
    b: JudgeOut | None = None
    tiebreak: JudgeOut | None = None


class MainstreamItem(BaseModel):
    outlet: str
    rating: str
    url: str
    published_at: str | None = None


class MainstreamCheck(BaseModel):
    label: Literal["What mainstream fact-checkers say"] = "What mainstream fact-checkers say"
    consensus: Consensus
    checks: list[MainstreamItem] = Field(default_factory=list)


class CheckRow(BaseModel):
    name: str
    passed: bool
    detail: str


class ClaimOut(BaseModel):
    text: str
    claim_type: Literal["fact", "opinion"]
    claim_category: str
    category_label: str = ""
    effort: str = ""
    verdict: Verdict | None = None
    strength_of_support: float = 0.0
    confidence: float = 0.0
    reasoning: str
    attribution: str | None = None
    needs_human_review: bool = False
    hypothesis_status: HypothesisStatus | None = None
    would_settle: str = ""
    checked_at: str = ""
    deterministic_checks: list[CheckRow] = Field(default_factory=list)
    tool: dict | None = None
    judges: JudgesOut | None = None
    sources: list[SourceOut] = Field(default_factory=list)
    mainstream_check: MainstreamCheck | None = None


class DraftAnswer(BaseModel):
    """A model draft. It is not evidence and has no weight in a verdict."""

    model: str
    text: str
    stance: Literal["yes", "no", "partly", "not_enough_evidence"] = "not_enough_evidence"
    correction: str = ""


class SynopsisCounts(BaseModel):
    supported: int = 0
    contradicted: int = 0
    conflicting: int = 0
    unverified: int = 0
    hypothesis: int = 0
    opinion: int = 0
    not_checked: int = 0


class SynopsisOut(BaseModel):
    """Plain-language bottom line. The rating is fixed from the claim verdicts, not from a model."""

    rating: Literal[
        "Accurate",
        "Mostly accurate",
        "Mixed",
        "Mostly inaccurate",
        "Inaccurate",
        "Not enough evidence",
    ]
    summary: str
    tldr: str = ""
    answer: Literal["yes", "no", "partly", "not_enough_evidence"] = "not_enough_evidence"
    caveats: list[str] = Field(default_factory=list)
    unchecked_count: int = 0
    counts: SynopsisCounts = Field(default_factory=SynopsisCounts)


class ReviewChange(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    claim: str
    from_: str = Field(alias="from")
    to: str
    reason: str


class ReviewOut(BaseModel):
    """Two-pass review. Empty when the review did not run."""

    passes: list[str] = Field(default_factory=list)
    changes: list[ReviewChange] = Field(default_factory=list)
    signed_off_by: str = ""


class UsageOut(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    retriever_cost_usd: float = 0.0
    credits_charged: float = 0.0
    tier: str = "free"
    models: dict[str, str] = Field(default_factory=dict)


class VerificationOut(BaseModel):
    verification_id: str
    created_at: str
    input_summary: str
    claims: list[ClaimOut]
    mode: Literal["question", "text"] = "text"
    draft_answer: DraftAnswer | None = None
    synopsis: SynopsisOut | None = None
    summary: str
    attribution: str
    disclaimer: str
    usage: UsageOut
    upgrade_url: str
    warnings: list[str] = Field(default_factory=list)
    review: ReviewOut = Field(default_factory=ReviewOut)


class VerifyIn(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    text: str | None = None
    claims: list[str] | None = None
    question: str | None = None
    async_job: bool = Field(default=False, alias="async")


DISCLAIMER = (
    f"Evidence summarized by {SERVICE_NAME}. Grounding reduces errors. "
    "It cannot prove that no better evidence exists, and a verdict is not proof."
)
