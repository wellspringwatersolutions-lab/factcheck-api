"""Internal records passed between pipeline stages. API models live in schemas.py."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractedClaim:
    text: str
    claim_type: str = "fact"
    claim_category: str = ""
    time_sensitive: bool = False
    scientific: bool = False
    attribution: str | None = None
    fragile: bool = False
    hypothesis: bool = False
    span: str = ""


@dataclass
class Evidence:
    url: str
    title: str
    snippet: str
    published_at: str | None = None
    provider: str = ""
    grade: str = "E"
    source_type: str = "secondary"
    stale: bool = False
    freshness: float = 1.0
    origin_id: str = ""
    syndicated_also: list[str] = field(default_factory=list)
    quote: str = ""
    content_sha256: str = ""
    url_status: int | None = None
    quote_match: bool | None = None
    number_match: bool | None = None
    date_match: bool | None = None
    entailment: str = ""
    page_text: str = ""
    used: bool = False
    doi: str = ""
    pmid: str = ""
    study_type: str = ""
    citation_count: int = 0
    influential_citations: int = 0
    open_access: bool = False
    venue: str = ""
    rank_score: float = 0.0


@dataclass
class JudgeVote:
    role: str
    model: str
    provider: str
    family: str
    verdict: str
    reasoning: str
    stances: dict[str, str] = field(default_factory=dict)
    ranked_urls: list[str] = field(default_factory=list)
    quotes: dict[str, str] = field(default_factory=dict)


@dataclass
class OutletCheck:
    outlet: str
    rating: str
    url: str
    published_at: str | None
    polarity: str


@dataclass
class MainstreamResult:
    consensus: str
    checks: list[OutletCheck]
    reasoning: str
    strength_of_support: float


@dataclass
class CitationCheck:
    cited: bool
    status: str
    note: str


@dataclass
class UsageMeter:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    retriever_cost_usd: float = 0.0
    retrieval_calls: int = 0
    models: dict[str, str] = field(default_factory=dict)

    def add_tokens(self, *, model: str, provider: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.estimated_cost_usd += cost
        self.models[provider] = model

    def add_usd(self, amount: float, *, retriever: bool = False) -> None:
        if amount <= 0:
            return
        self.estimated_cost_usd += amount
        if retriever:
            self.retriever_cost_usd += amount
