"""Keyless history and reference sources, plus optional archive catalogs.

Wikipedia is a secondary reference and shares one origin, so copies of it are not
independent. Crossref, the Internet Archive, and GDELT need no key. NARA, the
Smithsonian, DPLA, and Europeana are skipped when their keys are blank.
"""

from __future__ import annotations

import re

import httpx

from factcheck.references import user_agent
from factcheck.scholarly import RateLimiter
from factcheck.types import Evidence

_DOI = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_HTTP = re.compile(r"https?://[^\s)>\]]+")


def _get(client: httpx.Client, limiter: RateLimiter, url: str, *, params=None, headers=None) -> httpx.Response | None:
    limiter.acquire()
    try:
        response = client.get(url, params=params, headers=headers)
    except httpx.HTTPError:
        return None
    if response.status_code >= 400:
        return None
    return response


class WikipediaRetriever:
    """MediaWiki search plus a plain-text extract. Secondary reference, one origin."""

    name = "wikipedia"

    def __init__(self, email: str = "dev@localhost", client: httpx.Client | None = None, min_interval: float = 0.2):
        self.email = email
        self.client = client or httpx.Client(timeout=6.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 2) -> list[Evidence]:
        headers = {"User-Agent": user_agent(self.email), "Accept": "application/json"}
        found = _get(
            self.client,
            self.limiter,
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": min(limit, 3),
                "format": "json",
                "formatversion": "2",
            },
            headers=headers,
        )
        if found is None:
            self.warnings.append("wikipedia:http")
            return []
        rows = ((found.json().get("query") or {}).get("search")) or []
        titles = [row.get("title") for row in rows if row.get("title")]
        extracts: dict[str, str] = {}
        if titles:
            body = _get(
                self.client,
                self.limiter,
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": 1,
                    "explaintext": 1,
                    "exchars": 600,
                    "titles": "|".join(titles[:3]),
                    "format": "json",
                    "formatversion": "2",
                },
                headers=headers,
            )
            if body is not None:
                for page in ((body.json().get("query") or {}).get("pages")) or []:
                    extracts[page.get("title") or ""] = page.get("extract") or ""
        hits = []
        for row in rows:
            title = row.get("title") or ""
            text = extracts.get(title) or re.sub(r"<[^>]+>", " ", row.get("snippet") or "")
            text = re.sub(r"\s+", " ", text).strip()
            if not title or not text:
                continue
            slug = title.replace(" ", "_")
            hits.append(
                Evidence(
                    url=f"https://en.wikipedia.org/wiki/{slug}",
                    title=title,
                    snippet=text[:500],
                    page_text=text[:1200],
                    provider=self.name,
                    source_type="secondary",
                    grade="C",
                    origin_id="wikipedia.org",
                    study_type="reference",
                )
            )
        return hits[:limit]


class CrossrefRetriever:
    """Confirms a cited paper or book exists. Polite pool via mailto. No key."""

    name = "crossref"

    def __init__(self, email: str = "dev@localhost", client: httpx.Client | None = None, min_interval: float = 0.4):
        self.email = email or "dev@localhost"
        self.client = client or httpx.Client(timeout=6.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 3) -> list[Evidence]:
        doi = _DOI.search(query or "")
        headers = {"User-Agent": user_agent(self.email), "Accept": "application/json"}
        if doi:
            url = f"https://api.crossref.org/works/{doi.group(1)}"
            params = {"mailto": self.email}
        else:
            url = "https://api.crossref.org/works"
            params = {"query.bibliographic": query, "rows": min(limit, 5), "mailto": self.email}
        response = _get(self.client, self.limiter, url, params=params, headers=headers)
        if response is None:
            self.warnings.append("crossref:http")
            return []
        message = (response.json().get("message")) or {}
        items = message.get("items")
        if items is None:
            items = [message] if message.get("title") or message.get("DOI") else []
        hits = []
        for item in items:
            titles = item.get("title") or []
            title = titles[0] if titles else item.get("DOI") or ""
            doi_value = item.get("DOI") or ""
            abstract = re.sub(r"<[^>]+>", " ", item.get("abstract") or "")
            text = re.sub(r"\s+", " ", abstract).strip() or title
            year = None
            issued = ((item.get("issued") or {}).get("date-parts") or [[]])[0]
            if issued:
                year = str(issued[0])
            link = f"https://doi.org/{doi_value}" if doi_value else ""
            if not link or not title:
                continue
            hits.append(
                Evidence(
                    url=link,
                    title=title,
                    snippet=text[:500],
                    page_text=text[:1200],
                    published_at=year,
                    provider=self.name,
                    source_type="peer_reviewed",
                    grade="B",
                    doi=doi_value,
                    study_type="review",
                )
            )
        return hits[:limit]


class InternetArchiveRetriever:
    """Wayback for a cited URL, otherwise an archive.org metadata search. No key."""

    name = "internet_archive"

    def __init__(self, email: str = "dev@localhost", client: httpx.Client | None = None, min_interval: float = 0.5):
        self.email = email
        self.client = client or httpx.Client(timeout=6.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 3) -> list[Evidence]:
        headers = {"User-Agent": user_agent(self.email), "Accept": "application/json"}
        cited = _HTTP.search(query or "")
        if cited:
            response = _get(
                self.client,
                self.limiter,
                "https://archive.org/wayback/available",
                params={"url": cited.group(0)},
                headers=headers,
            )
            if response is None:
                self.warnings.append("wayback:http")
                return []
            closest = ((response.json().get("archived_snapshots") or {}).get("closest")) or {}
            if not closest.get("available") or not closest.get("url"):
                return []
            return [
                Evidence(
                    url=closest["url"],
                    title="Wayback Machine snapshot",
                    snippet=f"Archived copy of {cited.group(0)} at {closest.get('timestamp') or 'an earlier date'}.",
                    page_text=f"Archived copy of {cited.group(0)}.",
                    published_at=(closest.get("timestamp") or "")[:8],
                    provider=self.name,
                    source_type="secondary",
                    grade="C",
                    origin_id=cited.group(0),
                    study_type="archive_snapshot",
                )
            ]
        response = _get(
            self.client,
            self.limiter,
            "https://archive.org/advancedsearch.php",
            params={"q": query, "output": "json", "rows": min(limit, 5), "fl[]": "identifier,title,description,date"},
            headers=headers,
        )
        if response is None:
            self.warnings.append("archive:http")
            return []
        docs = ((response.json().get("response") or {}).get("docs")) or []
        hits = []
        for doc in docs:
            identifier = doc.get("identifier") or ""
            title = doc.get("title") or identifier
            if isinstance(title, list):
                title = title[0]
            description = doc.get("description") or ""
            if isinstance(description, list):
                description = " ".join(str(part) for part in description)
            text = re.sub(r"\s+", " ", str(description)).strip() or str(title)
            if not identifier:
                continue
            hits.append(
                Evidence(
                    url=f"https://archive.org/details/{identifier}",
                    title=str(title),
                    snippet=text[:500],
                    page_text=text[:1200],
                    published_at=str(doc.get("date") or "")[:10] or None,
                    provider=self.name,
                    source_type="primary_source",
                    grade="B",
                    origin_id="archive.org",
                    study_type="historical_text",
                )
            )
        return hits[:limit]


class GdeltRetriever:
    """DOC 2.0 article list. No key. One request, then the limiter waits five seconds."""

    name = "gdelt"

    def __init__(self, client: httpx.Client | None = None, min_interval: float = 5.0):
        self.client = client or httpx.Client(timeout=6.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 3) -> list[Evidence]:
        response = _get(
            self.client,
            self.limiter,
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": query, "mode": "ArtList", "format": "json", "maxrecords": min(limit, 5), "timespan": "3m"},
        )
        if response is None:
            self.warnings.append("gdelt:http")
            return []
        hits = []
        for article in response.json().get("articles") or []:
            url = article.get("url") or ""
            title = article.get("title") or ""
            if not url or not title:
                continue
            hits.append(
                Evidence(
                    url=url,
                    title=title,
                    snippet=title,
                    page_text=title,
                    published_at=article.get("seendate"),
                    provider=self.name,
                    source_type="secondary",
                    grade="C",
                    study_type="news",
                )
            )
        return hits[:limit]


class _KeyedRetriever:
    name = "keyed"
    endpoint = ""

    def __init__(self, api_key: str = "", client: httpx.Client | None = None, min_interval: float = 0.3):
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=6.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 3) -> list[Evidence]:
        if not self.api_key:
            return []
        return self._search(query, limit=limit)

    def _search(self, query: str, *, limit: int) -> list[Evidence]:
        return []


class NaraRetriever(_KeyedRetriever):
    name = "nara"

    def _search(self, query: str, *, limit: int) -> list[Evidence]:
        response = _get(
            self.client,
            self.limiter,
            "https://catalog.archives.gov/api/v2/records/search",
            params={"q": query, "limit": min(limit, 5)},
            headers={"x-api-key": self.api_key, "Accept": "application/json"},
        )
        if response is None:
            self.warnings.append("nara:http")
            return []
        payload = response.json()
        rows = payload.get("result") or []
        if not rows:
            hits = (((payload.get("body") or {}).get("hits") or {}).get("hits")) or []
            rows = [hit.get("_source") or hit for hit in hits]
        found = []
        for row in rows:
            record = row.get("record") or row
            title = record.get("title") or ""
            naid = str(record.get("naId") or row.get("naId") or row.get("_id") or "")
            text = record.get("scopeAndContentNote") or record.get("description") or title
            if not title or not naid:
                continue
            found.append(
                Evidence(
                    url=f"https://catalog.archives.gov/id/{naid}",
                    title=title,
                    snippet=str(text)[:500],
                    page_text=str(text)[:1200],
                    provider=self.name,
                    source_type="primary_source",
                    grade="A",
                    study_type="catalog_record",
                )
            )
        return found[:limit]


class SmithsonianRetriever(_KeyedRetriever):
    name = "smithsonian"

    def _search(self, query: str, *, limit: int) -> list[Evidence]:
        response = _get(
            self.client,
            self.limiter,
            "https://api.si.edu/openaccess/api/v1.0/search",
            params={"q": query, "rows": min(limit, 5), "api_key": self.api_key},
        )
        if response is None:
            self.warnings.append("smithsonian:http")
            return []
        rows = ((response.json().get("response") or {}).get("rows")) or []
        hits = []
        for row in rows:
            content = row.get("content") or {}
            descriptive = content.get("descriptiveNonRepeating") or {}
            title_field = descriptive.get("title") or {}
            title = title_field.get("content") if isinstance(title_field, dict) else row.get("title") or ""
            notes = ((content.get("freetext") or {}).get("notes")) or []
            text = " ".join(note.get("content") or "" for note in notes if isinstance(note, dict)) or title
            link = descriptive.get("record_link") or ""
            if not title or not link:
                continue
            hits.append(
                Evidence(
                    url=link,
                    title=title,
                    snippet=text[:500],
                    page_text=text[:1200],
                    provider=self.name,
                    source_type="primary_source",
                    grade="A",
                    study_type="museum_record",
                )
            )
        return hits[:limit]


class DplaRetriever(_KeyedRetriever):
    name = "dpla"

    def _search(self, query: str, *, limit: int) -> list[Evidence]:
        response = _get(
            self.client,
            self.limiter,
            "https://api.dp.la/v2/items",
            params={"q": query, "page_size": min(limit, 5), "api_key": self.api_key},
        )
        if response is None:
            self.warnings.append("dpla:http")
            return []
        hits = []
        for doc in response.json().get("docs") or []:
            resource = doc.get("sourceResource") or {}
            title = resource.get("title") or ""
            if isinstance(title, list):
                title = title[0]
            description = resource.get("description") or ""
            if isinstance(description, list):
                description = " ".join(str(part) for part in description)
            url = doc.get("isShownAt") or ""
            if isinstance(url, list):
                url = url[0] if url else ""
            item_id = doc.get("id") or ""
            if not url and item_id:
                url = f"https://dp.la/item/{item_id}"
            if not title or not url:
                continue
            hits.append(
                Evidence(
                    url=str(url),
                    title=str(title),
                    snippet=str(description)[:500] or str(title),
                    page_text=(str(description) or str(title))[:1200],
                    provider=self.name,
                    source_type="primary_source",
                    grade="B",
                    study_type="library_record",
                )
            )
        return hits[:limit]


class EuropeanaRetriever(_KeyedRetriever):
    name = "europeana"

    def _search(self, query: str, *, limit: int) -> list[Evidence]:
        response = _get(
            self.client,
            self.limiter,
            "https://api.europeana.eu/record/v2/search.json",
            params={"query": query, "rows": min(limit, 5)},
            headers={"X-Api-Key": self.api_key, "Accept": "application/json"},
        )
        if response is None:
            self.warnings.append("europeana:http")
            return []
        hits = []
        for item in response.json().get("items") or []:
            title = item.get("title") or ""
            if isinstance(title, list):
                title = title[0] if title else ""
            description = item.get("dcDescription") or item.get("dcDescriptionLangAware") or ""
            if isinstance(description, dict):
                description = " ".join(str(part) for parts in description.values() for part in (parts if isinstance(parts, list) else [parts]))
            if isinstance(description, list):
                description = " ".join(str(part) for part in description)
            guid = item.get("guid") or item.get("link") or ""
            if not title or not guid:
                continue
            hits.append(
                Evidence(
                    url=str(guid),
                    title=str(title),
                    snippet=str(description)[:500] or str(title),
                    page_text=(str(description) or str(title))[:1200],
                    provider=self.name,
                    source_type="primary_source",
                    grade="B",
                    study_type="cultural_heritage",
                )
            )
        return hits[:limit]
