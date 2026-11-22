"""Pluggable retrieval. Desearch is primary; academic APIs are used for scientific claims."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

import httpx

logger = logging.getLogger("factcheck")

from factcheck.parallel import SEARCH_SLOTS, parallel_map
from factcheck.request_ctx import CLAIM_CALL_CAP, bind_context, note_cost, reserve_retrieval
from factcheck.types import Evidence

_DATE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def _cost_header(response: httpx.Response) -> float:
    raw = response.headers.get("X-Desearch-Cost-Usd") or response.headers.get("x-desearch-cost-usd") or "0"
    try:
        return float(raw)
    except ValueError:
        return 0.0


class _BudgetedClient:
    """Every Desearch HTTP call, including a copied call path, shares one counter."""

    def __init__(self, client: httpx.Client, owner: "DesearchRetriever"):
        self._client = client
        self._owner = owner

    def request(self, method: str, url: str, **kwargs):
        if not reserve_retrieval(getattr(self._owner, "_budget", None)):
            logger.warning("desearch call cap reached before %s", url)
            return httpx.Response(429, request=httpx.Request(method, url), json={"error": "call_cap"})
        self._owner.retrieval_calls += 1
        return self._client.request(method, url, **kwargs)

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)


class DesearchRetriever:
    """GET /web for search and GET /web/extract for page text. /web/crawl is the legacy alias."""

    name = "desearch"
    scientific_only = False

    def __init__(self, api_key: str, base_url: str = "https://api.desearch.ai", client: httpx.Client | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._http = client or httpx.Client(timeout=8.0)
        self.client = _BudgetedClient(self._http, self)
        self.cost_usd = 0.0
        self.retrieval_calls = 0
        self._budget = None

    def bind_budget(self, state) -> None:
        self._budget = state

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self.api_key}

    def _exchange(self, method: str, url: str, *, budget, **kwargs):
        """One attempt, then one retry on a timeout or a 5xx. Each attempt counts toward the cap."""
        if budget is not None:
            self._budget = budget

        def once():
            try:
                return self.client.request(method, url, **kwargs)
            except httpx.TimeoutException:
                logger.warning("desearch timeout method=%s url=%s", method, url)
                return "timeout"

        response = once()
        retry = response == "timeout" or (response is not None and getattr(response, "status_code", 0) >= 500)
        if retry:
            status = "timeout" if response == "timeout" else response.status_code
            logger.warning("desearch retry after status=%s url=%s", status, url)
            response = once()
        if response is None or response == "timeout":
            return None
        if response.status_code >= 400:
            logger.warning("desearch status=%s url=%s", response.status_code, url)
            return None
        return response

    def _rows(self, path: str, query: str, limit: int, window: dict | None, budget=None) -> list[Evidence]:
        params = {"query": query, "start": 0}
        response = self._exchange("GET", f"{self.base_url}{path}", budget=budget, params=params, headers=self._headers())
        if response is None:
            return []
        payload = response.json()
        header_cost = _cost_header(response)
        body_cost = float(payload.get("cost_usd") or 0) if isinstance(payload, dict) else 0.0
        spent = header_cost or body_cost
        self.cost_usd += spent
        note_cost(spent)
        rows = payload.get("data") or [] if isinstance(payload, dict) else payload or []
        hits = []
        for row in rows[:limit]:
            hits.append(
                Evidence(
                    url=row.get("link") or row.get("url") or "",
                    title=row.get("title") or "",
                    snippet=row.get("snippet") or row.get("highlight") or "",
                    published_at=row.get("published_at") or row.get("published_date") or row.get("date"),
                    provider="desearch",
                )
            )
        return [hit for hit in hits if hit.url]

    def search(self, query: str, *, limit: int = 6) -> list[Evidence]:
        """Keyword web results. Dated search uses POST /search."""
        return self._rows("/web", query, limit, None)

    def _post_search(self, query: str, limit: int, window: dict | None, budget=None) -> list[Evidence]:
        """POST /search. The date window is not sent: Desearch would drop undated pages."""
        del window
        body: dict = {"query": query, "count": min(10, max(limit, 1)), "highlights": True}
        response = self._exchange("POST", f"{self.base_url}/search", budget=budget, json=body, headers=self._headers())
        if response is None:
            return []
        payload = response.json()
        header_cost = _cost_header(response)
        body_cost = float(payload.get("cost_usd") or 0) if isinstance(payload, dict) else 0.0
        spent = header_cost or body_cost
        self.cost_usd += spent
        note_cost(spent)
        rows = payload.get("results") or [] if isinstance(payload, dict) else []
        hits = []
        for row in rows[:limit]:
            highlights = row.get("highlights") or []
            if isinstance(highlights, list):
                snippet = " ".join(str(item) for item in highlights if item)
            else:
                snippet = str(highlights or row.get("snippet") or "")
            hits.append(
                Evidence(
                    url=row.get("url") or "",
                    title=row.get("title") or "",
                    snippet=snippet,
                    published_at=row.get("published_date") or row.get("published_at"),
                    provider="desearch",
                )
            )
        return [hit for hit in hits if hit.url]

    def _one_query(self, query: str, limit: int, window: dict | None, budget) -> list[Evidence]:
        """One Desearch call. The shared slot cap is about 8 in flight across claims."""
        with SEARCH_SLOTS:
            try:
                if window:
                    return self._post_search(query, min(limit, 10), window, budget)
                return self._rows("/web", query, min(limit, 10), None, budget)
            except Exception as exc:  # noqa: BLE001
                logger.warning("desearch query failed: %s", exc.__class__.__name__)
                return []

    def search_web_and_news(
        self,
        queries: list[str],
        *,
        limit: int = 10,
        window: dict | None = None,
        budget=None,
        stop_when=None,
    ) -> list[Evidence]:
        """A dated claim uses POST /search. Anything else uses GET /web. Never both, never GET /news.

        The first query runs alone. If it already settles the claim, the rest are not sent.
        Otherwise the remaining queries run together.
        """
        cleaned = []
        for query in queries:
            text = " ".join((query or "").split())
            if text and text not in cleaned:
                cleaned.append(text)
        hits: list[Evidence] = []
        seen: set[str] = set()

        def absorb(query: str, batch: list[Evidence]) -> None:
            logger.warning("desearch query=%r returned=%d dated=%s", query, len(batch), bool(window))
            for hit in batch:
                key = hit.url.split("#")[0].rstrip("/")
                if not key or key in seen:
                    continue
                seen.add(key)
                hits.append(hit)

        def settled() -> bool:
            if stop_when is None:
                return False
            try:
                return bool(stop_when(list(hits)))
            except Exception as exc:  # noqa: BLE001 - a stop check must not drop the hits we have
                logger.warning("desearch early stop check failed %s", exc.__class__.__name__)
                return False

        if not cleaned:
            return []
        if CLAIM_CALL_CAP <= 0:
            return []
        absorb(cleaned[0], self._one_query(cleaned[0], limit, window, budget))
        if settled():
            logger.warning("desearch early stop after 1 query")
            return hits[:10]
        rest = cleaned[1:CLAIM_CALL_CAP]
        if rest:
            batches = parallel_map(
                [lambda query=query: self._one_query(query, limit, window, budget) for query in rest],
                limit=8,
            )
            for query, batch in zip(rest, batches):
                absorb(query, batch or [])
                if settled():
                    logger.warning("desearch early stop after query=%r", query)
                    break
        return hits[:10]

    def search_fact_checkers(self, claim: str) -> list[Evidence]:
        """One web query limited to the four fact-checker hosts. Counts toward the 60-call cap."""
        from factcheck.cleanup import focus_query

        focus = focus_query(claim) or " ".join((claim or "").split())
        query = (
            f"{focus} site:snopes.com OR site:politifact.com OR site:factcheck.org "
            "OR site:apnews.com ap-fact-check"
        )
        logger.warning("fact-checker fallback query=%r", query[:180])
        return self._one_query(" ".join(query.split()), 5, None, self._budget)

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
        self.retrieval_calls = 0

    def bind_budget(self, state) -> None:
        for provider in self.providers:
            bind = getattr(provider, "bind_budget", None)
            if bind is not None:
                bind(state)

    @property
    def cost_usd(self) -> float:
        return sum(getattr(provider, "cost_usd", 0.0) for provider in self.providers)

    def search(
        self,
        query: str,
        *,
        limit: int = 6,
        scientific: bool = False,
        categories: list[str] | None = None,
        extra_queries: list[str] | None = None,
        window: dict | None = None,
        broad: bool = False,
        budget=None,
        stop_when=None,
    ) -> list[Evidence]:
        jobs = []
        for provider in self.providers:
            if broad and hasattr(provider, "search_web_and_news"):
                queries = [query, *(extra_queries or [])]
                jobs.append(
                    lambda provider=provider, queries=queries: provider.search_web_and_news(
                        queries, limit=limit, window=window, budget=budget, stop_when=stop_when
                    )
                )
            else:
                jobs.append(lambda provider=provider: provider.search(query, limit=limit))
        if self.scholarly and (scientific or set(categories or []) & {"T2", "T3", "T4"}):
            jobs.append(lambda: self.scholarly.search(query, limit=limit, categories=categories or (["T2"] if scientific else [])))
        if self.references is not None:
            jobs.append(lambda: self.references.search(query, limit=limit, categories=categories))
        hits: list[Evidence] = []
        before_calls = [getattr(provider, "retrieval_calls", 0) for provider in self.providers]
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            futures = [pool.submit(bind_context(job)) for job in jobs]
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
        self.retrieval_calls += sum(
            getattr(provider, "retrieval_calls", 0) - previous
            for provider, previous in zip(self.providers, before_calls)
        )
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
        # Desearch missed the page. Harnyx is the next reader. Firecrawl is not wired.
        reader = getattr(self, "harnyx", None)
        if reader is None:
            return ""
        try:
            text = reader.read(evidence.url)
        except Exception as exc:  # noqa: BLE001 - a reader outage leaves the snippet in place
            logger.warning("harnyx read failed %s", exc.__class__.__name__)
            return ""
        return (text or "")[:4000]


class CorpusRetriever:
    """Closed corpus used by the evaluation harness."""

    def __init__(self, documents: list[Evidence]):
        self.documents = documents
        self.cost_usd = 0.0
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 5, scientific: bool = False, categories: list[str] | None = None, **_ignored) -> list[Evidence]:
        from factcheck.textutil import overlap

        ranked = sorted(self.documents, key=lambda doc: overlap(query, f"{doc.title} {doc.snippet}"), reverse=True)
        return [doc for doc in ranked if overlap(query, f"{doc.title} {doc.snippet}") > 0][:limit]


class EmptyRetriever:
    cost_usd = 0.0
    warnings: list[str] = []

    def search(self, query: str, *, limit: int = 6, scientific: bool = False, categories: list[str] | None = None, **_ignored) -> list[Evidence]:
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
