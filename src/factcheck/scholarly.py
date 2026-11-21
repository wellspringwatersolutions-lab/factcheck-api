"""OpenAlex, Semantic Scholar, PubMed, and Europe PMC for T2–T4 claims.

Searches are mocked in tests. A 429 or a timeout returns no hits instead of failing the check.
"""

from __future__ import annotations

import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from xml.etree import ElementTree

import httpx

from factcheck.types import Evidence

STUDY_RANK = {"systematic_review": 3, "rct": 2, "observational": 1, "review": 1, "other": 0}
_BIOMED = re.compile(
    r"\b(trial|patient|clinical|vaccine|disease|cancer|drug|therapy|dosage|placebo|"
    r"cohort|medline|pubmed|symptom|diagnosis|treatment|randomized|randomised)\b",
    re.I,
)


class RateLimiter:
    """Spaces calls. Interval 0 disables the wait, which tests use."""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
            self._next = time.monotonic() + self.interval


def study_type_of(*parts: str) -> str:
    blob = " ".join(part for part in parts if part).lower()
    if "meta-analysis" in blob or "meta analysis" in blob or "systematic review" in blob:
        return "systematic_review"
    if "randomized" in blob or "randomised" in blob or "clinical trial" in blob or "rct" in blob:
        return "rct"
    if "cohort" in blob or "observational" in blob or "case-control" in blob:
        return "observational"
    if "review" in blob:
        return "review"
    return "other"


def rank_score(study_type: str, citations: int, influential: int, open_access: bool) -> float:
    return STUDY_RANK.get(study_type, 0) * 1000 + math.log1p(max(citations, 0)) + influential / 10 + (0.5 if open_access else 0)


def abstract_from_inverted_index(index: dict | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, spots in index.items():
        for spot in spots or []:
            positions.append((int(spot), word))
    positions.sort()
    return " ".join(word for _, word in positions)


def paper_identity(evidence: Evidence) -> str:
    if evidence.doi:
        return "doi:" + evidence.doi.lower().removeprefix("https://doi.org/")
    if evidence.pmid:
        return "pmid:" + evidence.pmid
    return "url:" + evidence.url.split("#")[0].rstrip("/")


def dedupe_papers(hits: list[Evidence]) -> list[Evidence]:
    """Keep one record per DOI or PMID, preferring the higher-ranked copy and the longer text."""
    best: dict[str, Evidence] = {}
    for hit in hits:
        key = paper_identity(hit)
        current = best.get(key)
        if current is None:
            best[key] = hit
            continue
        longer = hit if len(hit.page_text or hit.snippet) > len(current.page_text or current.snippet) else current
        winner = hit if hit.rank_score > current.rank_score else current
        if longer is not winner and len(longer.page_text or longer.snippet) > len(winner.page_text or winner.snippet):
            winner.page_text = longer.page_text or longer.snippet
            winner.snippet = (winner.page_text or winner.snippet)[:1500]
        if not winner.doi:
            winner.doi = longer.doi or hit.doi or current.doi
        if not winner.pmid:
            winner.pmid = longer.pmid or hit.pmid or current.pmid
        best[key] = winner
    ranked = sorted(best.values(), key=lambda item: item.rank_score, reverse=True)
    return ranked


def _guarded_get(client: httpx.Client, limiter: RateLimiter, url: str, **kwargs) -> tuple[httpx.Response | None, str]:
    limiter.acquire()
    try:
        response = client.get(url, **kwargs)
    except httpx.HTTPError as exc:
        return None, exc.__class__.__name__
    if response.status_code == 429:
        return None, "rate_limited"
    if response.status_code >= 400:
        return None, f"http_{response.status_code}"
    return response, ""


def _paper(
    *,
    url: str,
    title: str,
    text: str,
    year: str | None,
    provider: str,
    doi: str = "",
    pmid: str = "",
    study_type: str = "other",
    citations: int = 0,
    influential: int = 0,
    open_access: bool = False,
    venue: str = "",
) -> Evidence:
    body = (text or "")[:4000]
    grade = {"systematic_review": "A", "rct": "B"}.get(study_type, "B")
    return Evidence(
        url=url,
        title=title,
        snippet=body[:1500],
        page_text=body,
        published_at=year,
        provider=provider,
        grade=grade,
        source_type="peer_reviewed",
        study_type=study_type,
        doi=doi.lower().removeprefix("https://doi.org/") if doi else "",
        pmid=str(pmid or ""),
        citation_count=citations,
        influential_citations=influential,
        open_access=open_access,
        venue=venue,
        rank_score=rank_score(study_type, citations, influential, open_access),
    )


class OpenAlexRetriever:
    name = "openalex"
    scientific_only = True

    def __init__(self, email: str = "", api_key: str = "", client: httpx.Client | None = None, min_interval: float = 0.2):
        self.email = email
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=8.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []
        self.cost_usd = 0.0

    def search(self, query: str, *, limit: int = 5) -> list[Evidence]:
        if not self.email:
            self.warnings.append("openalex:missing_email")
            return []
        params = {
            "search": query,
            "per-page": limit,
            "mailto": self.email,
            "select": "id,doi,display_name,publication_year,type,cited_by_count,open_access,primary_location,abstract_inverted_index,ids",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        response, problem = _guarded_get(self.client, self.limiter, "https://api.openalex.org/works", params=params)
        if response is None:
            self.warnings.append(f"openalex:{problem}")
            return []
        hits = []
        for work in response.json().get("results") or []:
            location = (work.get("primary_location") or {}).get("source") or {}
            ids = work.get("ids") or {}
            venue = location.get("display_name") or ""
            kind = study_type_of(work.get("display_name") or "", work.get("type") or "", venue)
            oa = work.get("open_access") or {}
            text = abstract_from_inverted_index(work.get("abstract_inverted_index"))
            doi = work.get("doi") or ids.get("doi") or ""
            pmid = str(ids.get("pmid") or "").rsplit("/", 1)[-1]
            url = oa.get("oa_url") or doi or work.get("id") or ""
            hits.append(
                _paper(
                    url=url,
                    title=work.get("display_name") or "",
                    text=text,
                    year=str(work["publication_year"]) if work.get("publication_year") else None,
                    provider=self.name,
                    doi=doi,
                    pmid=pmid,
                    study_type=kind,
                    citations=int(work.get("cited_by_count") or 0),
                    open_access=bool(oa.get("is_oa")),
                    venue=venue,
                )
            )
        return [hit for hit in hits if hit.url]


class SemanticScholarRetriever:
    name = "semantic_scholar"
    scientific_only = True

    def __init__(self, api_key: str = "", client: httpx.Client | None = None, min_interval: float = 1.0):
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=8.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []
        self.cost_usd = 0.0

    def search(self, query: str, *, limit: int = 4) -> list[Evidence]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        response, problem = _guarded_get(
            self.client,
            self.limiter,
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,url,year,externalIds,tldr,influentialCitationCount,openAccessPdf,publicationVenue,publicationTypes",
            },
            headers=headers,
        )
        if response is None:
            self.warnings.append(f"semantic_scholar:{problem}")
            return []
        hits = []
        for paper in response.json().get("data") or []:
            external = paper.get("externalIds") or {}
            tldr = (paper.get("tldr") or {}).get("text") or ""
            abstract = paper.get("abstract") or ""
            text = abstract or tldr
            if tldr and tldr not in text:
                text = f"{text}\n{tldr}".strip()
            paper_id = paper.get("paperId") or ""
            oa = paper.get("openAccessPdf") or {}
            venue = (paper.get("publicationVenue") or {}).get("name") or ""
            types = " ".join(paper.get("publicationTypes") or [])
            kind = study_type_of(paper.get("title") or "", types, venue)
            hits.append(
                _paper(
                    url=paper.get("url") or oa.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""),
                    title=paper.get("title") or "",
                    text=text,
                    year=str(paper["year"]) if paper.get("year") else None,
                    provider=self.name,
                    doi=external.get("DOI") or "",
                    pmid=str(external.get("PubMed") or ""),
                    study_type=kind,
                    citations=int(paper.get("citationCount") or 0),
                    influential=int(paper.get("influentialCitationCount") or 0),
                    open_access=bool(oa.get("url")),
                    venue=venue,
                )
            )
        return [hit for hit in hits if hit.url]


class PubMedRetriever:
    name = "pubmed"
    scientific_only = True

    def __init__(
        self,
        email: str = "dev@localhost",
        api_key: str = "",
        client: httpx.Client | None = None,
        min_interval: float | None = None,
    ):
        self.email = email
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=8.0)
        self.limiter = RateLimiter(0.1 if api_key else 0.4 if min_interval is None else min_interval)
        self.warnings: list[str] = []
        self.cost_usd = 0.0

    def _params(self, extra: dict) -> dict:
        params = {"tool": "factcheck", "email": self.email, **extra}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(self, query: str, *, limit: int = 3) -> list[Evidence]:
        found, problem = _guarded_get(
            self.client,
            self.limiter,
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=self._params({"db": "pubmed", "retmode": "json", "retmax": limit, "term": query}),
        )
        if found is None:
            self.warnings.append(f"pubmed:{problem}")
            return []
        ids = ((found.json().get("esearchresult") or {}).get("idlist")) or []
        if not ids:
            return []
        fetched, problem = _guarded_get(
            self.client,
            self.limiter,
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=self._params({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}),
        )
        if fetched is None:
            self.warnings.append(f"pubmed:{problem}")
            return []
        root = ElementTree.fromstring(fetched.text)
        hits = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID") or ""
            title = "".join(article.find(".//ArticleTitle").itertext()) if article.find(".//ArticleTitle") is not None else ""
            abstract = " ".join("".join(node.itertext()) for node in article.findall(".//AbstractText"))
            year = article.findtext(".//PubDate/Year")
            doi = ""
            for node in article.findall(".//ArticleId"):
                if (node.get("IdType") or "").lower() == "doi":
                    doi = (node.text or "").strip()
            journal = article.findtext(".//Journal/Title") or article.findtext(".//Title") or ""
            hits.append(
                _paper(
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    title=title,
                    text=abstract,
                    year=year,
                    provider=self.name,
                    doi=doi,
                    pmid=pmid,
                    study_type=study_type_of(title, abstract),
                    venue=journal,
                )
            )
        return [hit for hit in hits if hit.url]


class EuropePMCRetriever:
    name = "europepmc"
    scientific_only = True

    def __init__(self, client: httpx.Client | None = None, min_interval: float = 0.2):
        self.client = client or httpx.Client(timeout=8.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []
        self.cost_usd = 0.0

    def search(self, query: str, *, limit: int = 3) -> list[Evidence]:
        response, problem = _guarded_get(
            self.client,
            self.limiter,
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": limit, "resultType": "core"},
        )
        if response is None:
            self.warnings.append(f"europepmc:{problem}")
            return []
        hits = []
        for row in ((response.json().get("resultList") or {}).get("result")) or []:
            pmcid = row.get("pmcid") or ""
            text = row.get("abstractText") or ""
            if row.get("isOpenAccess") == "Y" and pmcid:
                full, _problem = _guarded_get(
                    self.client,
                    self.limiter,
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                )
                if full is not None and full.text:
                    plain = re.sub(r"<[^>]+>", " ", full.text)
                    plain = re.sub(r"\s+", " ", plain).strip()
                    if len(plain) > len(text):
                        text = plain[:4000]
            doi = row.get("doi") or ""
            pmid = str(row.get("pmid") or "")
            hits.append(
                _paper(
                    url=(f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else f"https://doi.org/{doi}" if doi else ""),
                    title=row.get("title") or "",
                    text=text,
                    year=str(row.get("pubYear") or "") or None,
                    provider=self.name,
                    doi=doi,
                    pmid=pmid,
                    study_type=study_type_of(row.get("title") or "", row.get("pubType") or ""),
                    citations=int(row.get("citedByCount") or 0),
                    open_access=row.get("isOpenAccess") == "Y",
                    venue=row.get("journalTitle") or "",
                )
            )
        return [hit for hit in hits if hit.url and hit.snippet]


class ScholarlyBundle:
    """Pick scholarly sources from the claim category and search them together."""

    def __init__(
        self,
        *,
        openalex_email: str = "",
        openalex_api_key: str = "",
        semantic_scholar_api_key: str = "",
        ncbi_email: str = "dev@localhost",
        ncbi_api_key: str = "",
        client: httpx.Client | None = None,
        cache_ttl: int = 86_400,
        timeout: float = 8.0,
    ):
        shared = client
        self.openalex = OpenAlexRetriever(openalex_email, openalex_api_key, client=shared, min_interval=0.2)
        self.semantic_scholar = SemanticScholarRetriever(semantic_scholar_api_key, client=shared, min_interval=1.0)
        self.pubmed = PubMedRetriever(ncbi_email, ncbi_api_key, client=shared)
        self.europepmc = EuropePMCRetriever(client=shared)
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, list[Evidence]]] = {}
        self.warnings: list[str] = []
        self.cost_usd = 0.0

    def providers_for(self, categories: list[str] | None, query: str) -> list:
        cats = set(categories or [])
        biomedical = "T4" in cats or bool(_BIOMED.search(query or ""))
        scientific = bool(cats & {"T2", "T3", "T4"}) or biomedical
        if not scientific and not cats:
            scientific = True
        chosen = []
        if scientific:
            chosen.extend([self.openalex, self.semantic_scholar])
        if biomedical:
            chosen.extend([self.pubmed, self.europepmc])
        return chosen

    def search(self, query: str, *, limit: int = 5, categories: list[str] | None = None) -> list[Evidence]:
        providers = self.providers_for(categories, query)
        if not providers:
            return []
        key = f"{','.join(sorted(item.name for item in providers))}|{query.strip().lower()}"
        cached = self._cache.get(key)
        if cached and cached[0] > time.time():
            return list(cached[1])[:limit]
        hits: list[Evidence] = []
        with ThreadPoolExecutor(max_workers=len(providers)) as pool:
            futures = {pool.submit(provider.search, query, limit=limit): provider for provider in providers}
            try:
                for future in as_completed(futures, timeout=self.timeout + 1):
                    provider = futures[future]
                    try:
                        hits.extend(future.result())
                    except Exception as exc:  # noqa: BLE001
                        self.warnings.append(f"{provider.name}:{exc.__class__.__name__}")
            except TimeoutError:
                self.warnings.append("scholarly:timeout")
                for future in futures:
                    future.cancel()
        for provider in providers:
            self.warnings.extend(provider.warnings)
            provider.warnings.clear()
        ranked = dedupe_papers(hits)
        self._cache[key] = (time.time() + self.cache_ttl, ranked)
        return ranked[:limit]
