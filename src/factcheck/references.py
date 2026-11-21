"""Keyless history sources, plus optional fact-check ratings and claim scoring.

Wikidata and Chronicling America need no key. Google Fact Check and ClaimBuster
are skipped when their keys are blank. Nothing here calls a paid search API.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

import httpx

from factcheck.scholarly import RateLimiter
from factcheck.types import Evidence

_HISTORY = re.compile(
    r"\b(born|birth|died|death|assassinated|president|king|queen|war|treaty|century|historic)\b",
    re.I,
)
_YEAR = re.compile(r"\b(1[7-9]\d{2}|196[0-3])\b")


def user_agent(email: str) -> str:
    contact = email or "dev@localhost"
    return f"factcheck/0.1 (mailto:{contact})"


def wants_reference_sources(categories: list[str] | None, query: str) -> bool:
    cats = set(categories or [])
    if "T5" in cats:
        return True
    text = query or ""
    return bool(_HISTORY.search(text) or _YEAR.search(text))


def wikidata_date_check(claim: str, passage: str) -> tuple[str, str] | None:
    """Match a claim's birth or death year to a Wikidata sentence. None if it does not apply."""
    for label, cue in (("date of birth", r"\b(born|birth)\b"), ("date of death", r"\b(died|death|assassinated)\b")):
        match = re.search(rf"{label}:\s*(\d{{4}})", passage or "", re.I)
        if not match or not re.search(cue, claim or "", re.I):
            continue
        years = re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", claim or "")
        if not years:
            continue
        sentence = match.group(0)
        if match.group(1) in years:
            return "entails", sentence
        return "contradicts", sentence
    return None


class WikidataRetriever:
    name = "wikidata"

    def __init__(self, email: str = "dev@localhost", client: httpx.Client | None = None, min_interval: float = 0.2):
        self.email = email
        self.client = client or httpx.Client(timeout=4.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 1) -> list[Evidence]:
        safe = (query or "").replace("\\", "").replace('"', "")[:80]
        if not safe:
            return []
        sparql = f"""
SELECT ?item ?itemLabel ?birth ?death WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:api "EntitySearch" .
    bd:serviceParam wikibase:endpoint "www.wikidata.org" .
    bd:serviceParam mwapi:search "{safe}" .
    bd:serviceParam mwapi:language "en" .
    ?item wikibase:apiOutputItem mwapi:item .
  }}
  OPTIONAL {{ ?item wdt:P569 ?birth . }}
  OPTIONAL {{ ?item wdt:P570 ?death . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}} LIMIT {max(1, limit)}
""".strip()
        self.limiter.acquire()
        try:
            response = self.client.get(
                "https://query.wikidata.org/sparql",
                params={"query": sparql, "format": "json"},
                headers={"User-Agent": user_agent(self.email), "Accept": "application/sparql-results+json"},
            )
        except httpx.HTTPError as exc:
            self.warnings.append(f"wikidata:{exc.__class__.__name__}")
            return []
        if response.status_code == 429:
            self.warnings.append("wikidata:rate_limited")
            return []
        if response.status_code >= 400:
            self.warnings.append(f"wikidata:http_{response.status_code}")
            return []
        hits = []
        for row in ((response.json().get("results") or {}).get("bindings")) or []:
            label = ((row.get("itemLabel") or {}).get("value")) or safe
            item = ((row.get("item") or {}).get("value")) or ""
            qid = item.rstrip("/").rsplit("/", 1)[-1]
            birth = (((row.get("birth") or {}).get("value")) or "")[:10]
            death = (((row.get("death") or {}).get("value")) or "")[:10]
            parts = [f"Wikidata: {label}."]
            if birth:
                parts.append(f"Date of birth: {birth}.")
            if death:
                parts.append(f"Date of death: {death}.")
            text = " ".join(parts)
            hits.append(
                Evidence(
                    url=f"https://www.wikidata.org/wiki/{qid}" if qid.startswith("Q") else item or "https://www.wikidata.org",
                    title=label,
                    snippet=text,
                    page_text=text,
                    provider=self.name,
                    source_type="structured",
                    grade="A",
                    study_type="structured_fact",
                )
            )
        return hits


class ChroniclingAmericaRetriever:
    """Historic newspaper pages via the loc.gov JSON API. OCR search snippets only."""

    name = "loc"
    endpoint = "https://www.loc.gov/collections/chronicling-america/"

    def __init__(self, email: str = "dev@localhost", client: httpx.Client | None = None, min_interval: float = 1.0):
        self.email = email
        self.client = client or httpx.Client(timeout=6.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 3) -> list[Evidence]:
        params = {"fo": "json", "dl": "page", "c": min(limit, 5), "qs": query, "ops": "AND"}
        year = _YEAR.search(query or "")
        if year:
            params["start_date"] = f"{year.group(1)}-01-01"
            params["end_date"] = f"{year.group(1)}-12-31"
        self.limiter.acquire()
        try:
            response = self.client.get(self.endpoint, params=params, headers={"User-Agent": user_agent(self.email)})
        except httpx.HTTPError as exc:
            self.warnings.append(f"loc:{exc.__class__.__name__}")
            return []
        if response.status_code == 429:
            self.warnings.append("loc:rate_limited")
            return []
        if response.status_code >= 400:
            self.warnings.append(f"loc:http_{response.status_code}")
            return []
        hits = []
        for row in response.json().get("results") or []:
            description = row.get("description") or row.get("excerpt") or row.get("ocr") or ""
            if isinstance(description, list):
                description = " ".join(str(part) for part in description)
            snippet = re.sub(r"\s+", " ", str(description)).strip()[:500]
            if not snippet:
                continue
            url = row.get("url") or row.get("id") or ""
            hits.append(
                Evidence(
                    url=str(url),
                    title=row.get("title") or "Chronicling America",
                    snippet=snippet,
                    page_text=snippet,
                    published_at=row.get("date"),
                    provider=self.name,
                    source_type="primary_source",
                    grade="C",
                    study_type="newspaper_ocr",
                )
            )
        return [hit for hit in hits if hit.url]


class ClaimBuster:
    """Optional check-worthiness scores. One short request, then give up."""

    endpoint = "https://idir.uta.edu/claimbuster/api/v2/score/text/"

    def __init__(self, api_key: str = "", client: httpx.Client | None = None):
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=2.0)

    def scores(self, texts: list[str]) -> dict[str, float]:
        if not self.api_key or not texts:
            return {}
        body = {"input_text": "\n".join(texts)}
        try:
            response = self.client.post(
                self.endpoint,
                json=body,
                headers={"x-api-key": self.api_key, "Accept": "application/json"},
            )
        except httpx.HTTPError:
            return {}
        if response.status_code >= 400:
            return {}
        payload = response.json()
        rows = payload.get("results") or payload.get("claim_scores") or []
        if isinstance(rows, dict):
            rows = [rows]
        found: dict[str, float] = {}
        for row in rows:
            text = str(row.get("text") or row.get("input_text") or "")
            try:
                score = float(row.get("score"))
            except (TypeError, ValueError):
                continue
            if text:
                found[text] = score
        if len(texts) == 1 and len(rows) == 1 and texts[0] not in found:
            try:
                found[texts[0]] = float(rows[0].get("score"))
            except (TypeError, ValueError, AttributeError):
                pass
        return found


class ReferenceBundle:
    def __init__(
        self,
        *,
        email: str = "dev@localhost",
        google=None,
        client: httpx.Client | None = None,
        timeout: float = 6.0,
    ):
        self.wikidata = WikidataRetriever(email, client=client, min_interval=0.2)
        self.loc = ChroniclingAmericaRetriever(email, client=client, min_interval=1.0)
        self.google = google
        self.timeout = timeout
        self.warnings: list[str] = []

    def providers_for(self, categories: list[str] | None, query: str) -> list:
        if not wants_reference_sources(categories, query):
            return []
        chosen = [self.wikidata, self.loc]
        if self.google is not None:
            chosen.append(self.google)
        return chosen

    def search(self, query: str, *, limit: int = 3, categories: list[str] | None = None) -> list[Evidence]:
        providers = self.providers_for(categories, query)
        if not providers:
            return []
        hits: list[Evidence] = []
        with ThreadPoolExecutor(max_workers=len(providers)) as pool:
            futures = {pool.submit(provider.search, query, limit=limit): provider for provider in providers}
            try:
                for future in as_completed(futures, timeout=self.timeout):
                    provider = futures[future]
                    try:
                        hits.extend(future.result())
                    except Exception as exc:  # noqa: BLE001
                        self.warnings.append(f"{provider.name}:{exc.__class__.__name__}")
            except TimeoutError:
                self.warnings.append("references:timeout")
        for provider in providers:
            self.warnings.extend(getattr(provider, "warnings", []))
            if hasattr(provider, "warnings"):
                provider.warnings.clear()
        return hits
