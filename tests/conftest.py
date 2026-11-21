from __future__ import annotations

from factcheck.billing import BillingGate, ClientIdentity
from factcheck.checks import MapUrlChecker
from factcheck.config import Settings
from factcheck.extract import HeuristicExtractor
from factcheck.judge import HeuristicRanker
from factcheck.mainstream import MainstreamChecker, build_result
from factcheck.storage import MemoryCache, Store, init_db, make_engine, make_session_factory
from factcheck.types import Evidence, OutletCheck, UsageMeter


class RecordingRetriever:
    def __init__(self, hits: list[Evidence] | None = None):
        self.hits = list(hits or [])
        self.queries: list[str] = []
        self.cost_usd = 0.0
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 6, scientific: bool = False, categories: list[str] | None = None, extra_queries=None, window=None, broad: bool = False, budget=None):
        from factcheck.request_ctx import note_retrieval

        self.queries.append(query)
        note_retrieval(1)
        self.extra_queries = [*getattr(self, "extra_queries", []), *(extra_queries or [])]
        self.windows = [*getattr(self, "windows", []), window]
        self.retrieval_calls = getattr(self, "retrieval_calls", 0) + 1
        if broad:
            self.retrieval_calls += len(extra_queries or [])
        return list(self.hits)[:limit]


class RecordingRanker:
    def __init__(self, role: str, model: str):
        self.role = role
        self.model = model
        self.family = model.split("/")[0]
        self.calls = 0
        self.meter = UsageMeter()

    def rank(self, claims, passages):
        from factcheck.judge import offline_label
        from factcheck.types import JudgeVote

        self.calls += 1
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
                    reasoning="Labeled passages. confidence 0.99 is not used.",
                    stances=stances,
                    ranked_urls=[url for url, label in stances.items() if label != "not_relevant"],
                    quotes=quotes,
                )
            )
        return votes


class FixedMainstream(MainstreamChecker):
    def __init__(self):
        super().__init__()
        self.seen = []

    def check(self, claims):
        self.seen.extend(claims)
        item = OutletCheck("Snopes", "False", "https://www.snopes.com/fact-check/example", "2024-01-02", "disputes")
        return [build_result([item]) for _ in claims]


def settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        database_url="sqlite://",
        llm_mode="mock",
        cache_backend="memory",
        free_daily_spend_cap_usd=20,
        free_check_reserve_usd=0.25,
        credits_per_check=1,
        public_base_url="http://127.0.0.1:8741",
    )
    base.update(overrides)
    return Settings(**base)


def make_service(hits=None, pages=None, retriever=None, **setting_overrides):
    from factcheck.orchestrator import VerificationService

    cfg = settings(**setting_overrides)
    engine = make_engine(cfg.database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    retriever = retriever or RecordingRetriever(hits)
    service = VerificationService(
        settings=cfg,
        extractor=HeuristicExtractor(),
        retriever=retriever,
        ranker_a=RecordingRanker("a", "mock-a/heuristic"),
        ranker_b=RecordingRanker("b", "mock-b/heuristic"),
        ranker_tie=RecordingRanker("tie", "mock-tie/heuristic"),
        mainstream=FixedMainstream(),
        store=Store(factory),
        cache=MemoryCache(),
        billing=BillingGate(factory, cfg),
        url_checker=MapUrlChecker(pages or {}),
        capital_fetcher=None,
    )
    return service, retriever, factory, cfg


def client(ip="203.0.113.5", fingerprint="agent-a", api_key=None) -> ClientIdentity:
    return ClientIdentity(ip=ip, user_agent="pytest", fingerprint_header=fingerprint, api_key=api_key)
