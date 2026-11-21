"""Pluggable retrieval. Desearch is primary; academic APIs are used for scientific claims."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

import httpx

from factcheck.types import Evidence

_DATE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def _cost_header(response: httpx.Response) -> float:
    raw = response.headers.get("X-Desearch-Cost-Usd") or response.headers.get("x-desearch-cost-usd") or "0"
    try:
        return float(raw)
    except ValueError:
        return 0.0


class DesearchRetriever:
    """GET /web for search and GET /web/extract for page text. /web/crawl is the legacy alias."""

    name = "desearch"
    scientific_only = False

    def __init__(self, api_key: str, base_url: str = "https://api.desearch.ai", client: httpx.Client | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=8.0)
        self.cost_usd = 0.0

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self.api_key}

    def search(self, query: str, *, limit: int = 6) -> list[Evidence]:
        response = self.client.get(
            f"{self.base_url}/web",
            params={"query": query, "start": 0},
            headers=self._headers(),
        )
        response.raise_for_status()
        payload = response.json()
        header_cost = _cost_header(response)
        body_cost = float(payload.get("cost_usd") or 0) if isinstance(payload, dict) else 0.0
        self.cost_usd += header_cost or body_cost
        if isinstance(payload, dict):
            rows = payload.get("data") or []
        else:
            rows = payload or []
        hits = []
        for row in rows[:limit]:
            hits.append(
                Evidence(
                    url=row.get("link") or row.get("url") or "",
                    title=row.get("title") or "",
                    snippet=row.get("snippet") or "",
                    provider="desearch",
                )
            )
        return [hit for hit in hits if hit.url]

    def extract(self, url: str) -> str:
        response = self.client.get(
            f"{self.base_url}/web/extract",
            params={"url": url, "format": "text"},
            headers=self._headers(),
        )
        self.cost_usd += _cost_header(response)
        if response.status_code >= 400:
            return ""
        return response.text or ""


class GoogleFactCheckRetriever:
    name = "google_factcheck"
    scientific_only = False

    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=15.0)
        self.cost_usd = 0.0

    def search_raw(self, query: str) -> dict:
        response = self.client.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": query, "key": self.api_key, "languageCode": "en", "pageSize": 8},
        )
        response.raise_for_status()
        return response.json()

    def search(self, query: str, *, limit: int = 6) -> list[Evidence]:
        payload = self.search_raw(query)
        hits = []
        for claim in payload.get("claims") or []:
            for review in (claim.get("claimReview") or [])[:limit]:
                publisher = (review.get("publisher") or {}).get("name") or "Fact-checker"
                rating = review.get("textualRating") or ""
                hits.append(
                    Evidence(
                        url=review.get("url") or "",
                        title=review.get("title") or publisher,
                        snippet=f"{publisher} rating: {rating}. {claim.get('text') or ''}".strip(),
                        page_text=f"{publisher} rating: {rating}.",
                        published_at=review.get("reviewDate"),
                        provider="google_factcheck",
                        source_type="fact_checker",
                        grade="B",
                        study_type="published_fact_check",
                    )
                )
        return [hit for hit in hits if hit.url][:limit]


from factcheck.scholarly import PubMedRetriever, SemanticScholarRetriever


class CompositeRetriever:
    def __init__(self, providers: list, scholarly=None, references=None):
        self.providers = [provider for provider in providers if not getattr(provider, "scientific_only", False)]
        self.scholarly = scholarly
        self.references = references
        self.warnings: list[str] = []
        self.timeout = 8.0

    @property
    def cost_usd(self) -> float:
        return sum(getattr(provider, "cost_usd", 0.0) for provider in self.providers)

    def search(self, query: str, *, limit: int = 6, scientific: bool = False, categories: list[str] | None = None) -> list[Evidence]:
        jobs = [lambda provider=provider: provider.search(query, limit=limit) for provider in self.providers]
        if self.scholarly and (scientific or set(categories or []) & {"T2", "T3", "T4"}):
            jobs.append(lambda: self.scholarly.search(query, limit=limit, categories=categories or (["T2"] if scientific else [])))
        if self.references is not None:
            jobs.append(lambda: self.references.search(query, limit=limit, categories=categories))
        hits: list[Evidence] = []
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            futures = [pool.submit(job) for job in jobs]
            try:
                for future in as_completed(futures, timeout=self.timeout):
                    try:
                        hits.extend(future.result())
                    except Exception as exc:  # noqa: BLE001
                        self.warnings.append(f"retriever:{exc.__class__.__name__}")
            except TimeoutError:
                self.warnings.append("retriever:timeout")
        if self.scholarly:
            self.warnings.extend(getattr(self.scholarly, "warnings", []))
        if self.references is not None:
            self.warnings.extend(getattr(self.references, "warnings", []))
        seen: set[str] = set()
        kept = []
        for hit in hits:
            key = hit.url.split("#")[0].rstrip("/")
            if not key or key in seen:
                continue
            seen.add(key)
            kept.append(hit)
        return kept

    def enrich(self, evidence: Evidence) -> str:
        for provider in self.providers:
            extract = getattr(provider, "extract", None)
            if extract is None:
                continue
            try:
                text = extract(evidence.url)
            except Exception:  # noqa: BLE001
                continue
            if text:
                found = _DATE.search(text)
                if found and not evidence.published_at:
                    evidence.published_at = found.group(1)
                return text[:4000]
        return ""


class CorpusRetriever:
    """Closed corpus used by the evaluation harness."""

    def __init__(self, documents: list[Evidence]):
        self.documents = documents
        self.cost_usd = 0.0
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 5, scientific: bool = False, categories: list[str] | None = None) -> list[Evidence]:
        from factcheck.textutil import overlap

        ranked = sorted(self.documents, key=lambda doc: overlap(query, f"{doc.title} {doc.snippet}"), reverse=True)
        return [doc for doc in ranked if overlap(query, f"{doc.title} {doc.snippet}") > 0][:limit]


class EmptyRetriever:
    cost_usd = 0.0
    warnings: list[str] = []

    def search(self, query: str, *, limit: int = 6, scientific: bool = False, categories: list[str] | None = None) -> list[Evidence]:
        return []


_CAPITAL = re.compile(r"\bcapital of ([A-Za-z][A-Za-z .'\-]{1,40}?) is ([A-Za-z][A-Za-z .'\-]{1,40})", re.I)


def lookup_capital(text: str, fetcher) -> dict | None:
    """Structured lookup against the REST Countries API. `fetcher(country) -> list[str] | None`."""
    match = _CAPITAL.search(text or "")
    if not match or fetcher is None:
        return None
    country = match.group(1).strip()
    claimed = match.group(2).strip().rstrip(".")
    try:
        capitals = fetcher(country)
    except Exception:  # noqa: BLE001
        return None
    if not capitals:
        return None
    actual = capitals[0]
    return {
        "tool": "restcountries",
        "matches": actual.lower() == claimed.lower(),
        "trace": f"REST Countries lists the capital of {country} as {actual}.",
        "actual": actual,
        "claimed": claimed,
    }


def restcountries_fetcher(client: httpx.Client | None = None):
    http = client or httpx.Client(timeout=10.0)

    def fetch(country: str) -> list[str]:
        response = http.get(f"https://restcountries.com/v3.1/name/{country}", params={"fields": "capital"})
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return []
        return list((rows[0] or {}).get("capital") or [])

    return fetch
