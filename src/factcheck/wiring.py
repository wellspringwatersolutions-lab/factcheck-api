"""Build the default service graph from settings."""

from __future__ import annotations

from factcheck.billing import BillingGate
from factcheck.checks import UrlChecker
from factcheck.config import Settings
from factcheck.extract import HeuristicExtractor, LiveExtractor
from factcheck.judge import HeuristicRanker, LiveRanker
from factcheck.llm import build_llm
from factcheck.mainstream import MainstreamChecker, from_google_payload
from factcheck.retrievers import (
    CompositeRetriever,
    DesearchRetriever,
    EmptyRetriever,
    GoogleFactCheckRetriever,
    restcountries_fetcher,
)
from factcheck.questions import AnswerDrafter
from factcheck.references import ClaimBuster, ReferenceBundle
from factcheck.synopsis import LLMSynopsis
from factcheck.scholarly import ScholarlyBundle
from factcheck.storage import Store, build_cache, init_db, make_engine, make_session_factory
from factcheck.types import UsageMeter


def build_service(settings: Settings):
    from factcheck.orchestrator import VerificationService

    engine = make_engine(settings.database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    if settings.bootstrap_api_key:
        from datetime import datetime, timezone
        from uuid import uuid4

        from sqlalchemy import select

        from factcheck.storage import ApiKeyRow, hash_key

        with factory() as session:
            exists = session.scalar(select(ApiKeyRow).where(ApiKeyRow.key_hash == hash_key(settings.bootstrap_api_key)))
            if exists is None:
                session.add(
                    ApiKeyRow(
                        id=str(uuid4()),
                        key_hash=hash_key(settings.bootstrap_api_key),
                        prefix=settings.bootstrap_api_key[:10],
                        label="bootstrap",
                        credits=settings.bootstrap_credits,
                        created_at=datetime.now(timezone.utc),
                    )
                )
                session.commit()

    meter = UsageMeter()
    client = None if settings.llm_mode == "mock" else build_llm(settings)
    drafter = None
    synopsizer = None
    if settings.llm_mode == "auto" and client is None:
        pass
    if client is None:
        extractor = HeuristicExtractor()
        ranker_a = HeuristicRanker("a", "mock-a/heuristic", meter)
        ranker_b = HeuristicRanker("b", "mock-b/heuristic", meter)
        ranker_tie = HeuristicRanker("tie", "mock-tie/heuristic", meter)
    else:
        extractor = LiveExtractor(client, settings.extract_model, meter, settings)
        ranker_a = LiveRanker(client, "a", settings.judge_a_model, meter, settings)
        ranker_b = LiveRanker(client, "b", settings.judge_b_model, meter, settings)
        ranker_tie = LiveRanker(client, "tie", settings.tiebreak_model, meter, settings)
        drafter = AnswerDrafter(client, settings.extract_model, meter, settings)
        synopsizer = LLMSynopsis(client, settings.extract_model, meter, settings)

    providers = []
    desearch = None
    google = None
    if settings.desearch_api_key:
        desearch = DesearchRetriever(settings.desearch_api_key, settings.desearch_base_url)
        providers.append(desearch)
    if settings.google_factcheck_api_key:
        google = GoogleFactCheckRetriever(settings.google_factcheck_api_key)
    scholarly = ScholarlyBundle(
        openalex_email=settings.openalex_email,
        openalex_api_key=settings.openalex_api_key,
        semantic_scholar_api_key=settings.semantic_scholar_api_key,
        ncbi_email=settings.ncbi_email,
        ncbi_api_key=settings.ncbi_api_key,
    )
    references = ReferenceBundle(email=settings.ncbi_email, google=google)
    claimbuster = ClaimBuster(settings.claimbuster_api_key) if settings.claimbuster_api_key else None
    retriever = CompositeRetriever(providers, scholarly=scholarly, references=references)

    def google_search(query: str):
        if google is None:
            return []
        try:
            return from_google_payload(google.search_raw(query), query)
        except Exception:
            return []

    def web_search(query: str):
        if desearch is None:
            return []
        try:
            return desearch.search(query, limit=6)
        except Exception:
            return []

    mainstream = MainstreamChecker(google_search if google else None, web_search if desearch else None)
    return VerificationService(
        settings=settings,
        extractor=extractor,
        retriever=retriever,
        ranker_a=ranker_a,
        ranker_b=ranker_b,
        ranker_tie=ranker_tie,
        mainstream=mainstream,
        store=Store(factory),
        cache=build_cache(settings),
        billing=BillingGate(factory, settings),
        url_checker=UrlChecker(timeout=min(12.0, settings.request_timeout_seconds)),
        capital_fetcher=restcountries_fetcher(),
        claimbuster=claimbuster,
        drafter=drafter,
        synopsizer=synopsizer,
    )
