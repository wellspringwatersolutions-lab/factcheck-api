"""Keyless history sources, plus optional fact-check ratings and claim scoring.

Wikidata and Chronicling America need no key. Google Fact Check and ClaimBuster
are skipped when their keys are blank. Nothing here calls a paid search API.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

import httpx

from factcheck.scholarly import RateLimiter
from factcheck.types import Evidence

_HISTORY = re.compile(
    r"\b(born|birth|died|death|assassinated|president|king|queen|war|treaty|century|historic)\b",
    re.I,
)
_YEAR = re.compile(r"\b(1[7-9]\d{2}|196[0-3])\b")
_DOI_IN = re.compile(r"\b10\.\d{4,9}/")
_HTTP_IN = re.compile(r"https?://")


def user_agent(email: str) -> str:
    contact = email or "dev@localhost"
    return f"factcheck/0.1 (mailto:{contact})"


_NEWS = re.compile(r"\b(today|this week|breaking|election|202[4-9]|203\d)\b", re.I)
_SCIENCE = re.compile(r"\b(doi|study|trial|journal|paper)\b", re.I)


def well_documented_fact(query: str) -> bool:
    """Biographies and offices that Wikipedia or Wikidata can settle without the slow archives."""
    return bool(
        re.search(
            r"\b(president|criminal|born|died|birth|death|founding father|commander|office held)\b",
            query or "",
            re.I,
        )
    )


_ACCUSATION = re.compile(r"\b(criminal|crimes?|convicted|conviction|felon|arrested|indicted|charged)\b", re.I)
_PERSON = re.compile(r"\b([A-Z][a-z]+(?:\s+(?:[A-Z]\.|[A-Z][a-z]+)){1,2})\b")
_NOT_NAME = {"The", "A", "An", "Was", "Is", "Did", "Has", "Were", "In", "On", "When", "Why", "How"}


def accusation_subject(query: str) -> str:
    """The person named in a criminality query, e.g. 'George Washington' from
    'George Washington criminal accusations historical records'. Empty when none."""
    text = query or ""
    if not _ACCUSATION.search(text):
        return ""
    for match in _PERSON.finditer(text):
        words = match.group(1).split()
        while words and words[0] in _NOT_NAME:
            words = words[1:]
        if len(words) >= 2:
            return " ".join(words)
    return ""


def wants_newspapers(query: str) -> bool:
    """Chronicling America is 1770–1963 newspapers. Skip later years and office facts."""
    text = query or ""
    if re.search(r"\b(19[7-9]\d|20\d{2})\b", text):
        return False
    if re.search(r"\b(newspaper|gazette|herald|chronicle|reported in)\b", text, re.I):
        return True
    if re.search(r"\b(president|commanding general|commander in chief|was born|date of birth)\b", text, re.I):
        return False
    return bool(_YEAR.search(text) or _HISTORY.search(text))


def wants_reference_sources(categories: list[str] | None, query: str) -> bool:
    cats = set(categories or [])
    if "T5" in cats:
        return True
    text = query or ""
    return bool(_HISTORY.search(text) or _YEAR.search(text))


_ORDINALS = {"first": "1", "1st": "1", "second": "2", "2nd": "2", "third": "3", "3rd": "3"}


def _role_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = lowered.replace("commander in chief", "commanding general")
    for word, number in _ORDINALS.items():
        lowered = re.sub(rf"\b{word}\b", f"ordinal {number}", lowered)
    return lowered


def encyclopedia_role_check(claim: str, passage: str) -> tuple[str, str] | None:
    """Match an office in a Wikipedia or encyclopedia lead sentence."""
    from factcheck.textutil import overlap

    if not re.search(r"\b(president|general|commander|king|queen|office)\b", claim or "", re.I):
        return None
    claim_role = _role_text(claim)
    from factcheck.textutil import split_sentences

    for sentence in split_sentences(passage or ""):
        sentence = sentence.strip()
        if not sentence or overlap(claim_role, _role_text(sentence)) < 0.45:
            continue
        record_ordinal = re.search(r"ordinal (\d+)", _role_text(sentence))
        claim_ordinal = re.search(r"ordinal (\d+)", claim_role)
        if claim_ordinal and record_ordinal and claim_ordinal.group(1) != record_ordinal.group(1):
            return "contradicts", sentence
        return "entails", sentence
    return None


def wikidata_office_check(claim: str, passage: str) -> tuple[str, str] | None:
    """Match a claimed office, rank, or event to a Wikidata sentence."""
    from factcheck.textutil import overlap, tokens

    claim_role = _role_text(claim)
    patterns = (
        r"Position held: ([^.]+)\.",
        r"Military rank: ([^.]+)\.",
        r"Participated in: ([^.]+)\.",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, passage or ""):
            sentence = match.group(0)
            role = _role_text(match.group(1))
            role_tokens = tokens(role)
            if not role_tokens or overlap(role, claim_role) < 0.6:
                continue
            record_ordinal = re.search(r"ordinal (\d+)", role)
            claim_ordinal = re.search(r"ordinal (\d+)", claim_role)
            if claim_ordinal and record_ordinal and claim_ordinal.group(1) != record_ordinal.group(1):
                return "contradicts", sentence
            if claim_ordinal and record_ordinal is None and "position held" in sentence.lower():
                continue
            return "entails", sentence
    return None


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
        self.client = client or httpx.Client(timeout=3.0)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []

    def _fetch(self, sparql: str):
        self.limiter.acquire()
        try:
            return self.client.get(
                "https://query.wikidata.org/sparql",
                params={"query": sparql, "format": "json"},
                headers={"User-Agent": user_agent(self.email), "Accept": "application/sparql-results+json"},
            )
        except httpx.HTTPError:
            return None

    def _wikipedia_lead(self, query: str) -> list[Evidence]:
        """Office fallback when Wikidata is down. One shared Wikipedia origin."""
        headers = {"User-Agent": user_agent(self.email), "Accept": "application/json"}
        try:
            found = self.client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 1,
                    "format": "json",
                    "formatversion": "2",
                },
                headers=headers,
            )
            title = (((found.json().get("query") or {}).get("search")) or [{}])[0].get("title") or ""
            if not title or found.status_code >= 400:
                return []
            extract = self.client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": 1,
                    "explaintext": 1,
                    "exchars": 600,
                    "titles": title,
                    "format": "json",
                    "formatversion": "2",
                },
                headers=headers,
            )
            pages = ((extract.json().get("query") or {}).get("pages")) or []
            text = (pages[0].get("extract") or "") if pages else ""
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            return []
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        return [
            Evidence(
                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                title=title,
                snippet=text[:500],
                page_text=text[:1200],
                provider="wikipedia",
                source_type="encyclopedia",
                grade="B",
                origin_id="wikipedia.org",
                study_type="reference",
            )
        ]

    def search(self, query: str, *, limit: int = 1) -> list[Evidence]:
        safe = (query or "").replace("\\", "").replace('"', "")[:80]
        if not safe:
            return []
        sparql = f"""
SELECT ?item ?itemLabel ?birth ?death ?officeLabel ?ordinal ?rankLabel ?conflictLabel WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:api "EntitySearch" .
    bd:serviceParam wikibase:endpoint "www.wikidata.org" .
    bd:serviceParam mwapi:search "{safe}" .
    bd:serviceParam mwapi:language "en" .
    ?item wikibase:apiOutputItem mwapi:item .
  }}
  OPTIONAL {{ ?item wdt:P569 ?birth . }}
  OPTIONAL {{ ?item wdt:P570 ?death . }}
  OPTIONAL {{
    ?item p:P39 ?stmt .
    ?stmt ps:P39 ?office .
    OPTIONAL {{ ?stmt pq:P1545 ?ordinal . }}
  }}
  OPTIONAL {{ ?item wdt:P410 ?rank . }}
  OPTIONAL {{ ?item wdt:P607 ?conflict . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}} LIMIT {max(12, limit)}
""".strip()
        response = self._fetch(sparql)
        if response is None or response.status_code >= 500:
            self.warnings.append("wikidata:retry")
            time.sleep(0 if self.limiter.interval <= 0 else 0.4)
            response = self._fetch(sparql)
        if response is None:
            self.warnings.append("wikidata:http")
            return self._wikipedia_lead(safe)
        if response.status_code == 429:
            self.warnings.append("wikidata:rate_limited")
            return self._wikipedia_lead(safe)
        if response.status_code >= 400:
            self.warnings.append(f"wikidata:http_{response.status_code}")
            return self._wikipedia_lead(safe)
        grouped: dict[str, dict] = {}
        for row in ((response.json().get("results") or {}).get("bindings")) or []:
            label = ((row.get("itemLabel") or {}).get("value")) or safe
            item = ((row.get("item") or {}).get("value")) or ""
            qid = item.rstrip("/").rsplit("/", 1)[-1] or label
            bucket = grouped.setdefault(
                qid,
                {"label": label, "item": item, "birth": "", "death": "", "offices": [], "ranks": [], "events": []},
            )
            birth = (((row.get("birth") or {}).get("value")) or "")[:10]
            death = (((row.get("death") or {}).get("value")) or "")[:10]
            if birth:
                bucket["birth"] = birth
            if death:
                bucket["death"] = death
            office = ((row.get("officeLabel") or {}).get("value")) or ""
            ordinal = ((row.get("ordinal") or {}).get("value")) or ""
            if office:
                entry = (office, ordinal)
                if entry not in bucket["offices"]:
                    bucket["offices"].append(entry)
            rank = ((row.get("rankLabel") or {}).get("value")) or ""
            if rank and rank not in bucket["ranks"]:
                bucket["ranks"].append(rank)
            conflict = ((row.get("conflictLabel") or {}).get("value")) or ""
            if conflict and conflict not in bucket["events"]:
                bucket["events"].append(conflict)
        hits = []
        for qid, bucket in list(grouped.items())[:limit]:
            parts = [f"Wikidata: {bucket['label']}."]
            if bucket["birth"]:
                parts.append(f"Date of birth: {bucket['birth']}.")
            if bucket["death"]:
                parts.append(f"Date of death: {bucket['death']}.")
            for office, ordinal in bucket["offices"]:
                suffix = f" (ordinal {ordinal})" if ordinal else ""
                parts.append(f"Position held: {office}{suffix}.")
            for rank in bucket["ranks"]:
                parts.append(f"Military rank: {rank}.")
            for event in bucket["events"]:
                parts.append(f"Participated in: {event}.")
            text = " ".join(parts)
            item = bucket["item"]
            hits.append(
                Evidence(
                    url=f"https://www.wikidata.org/wiki/{qid}" if qid.startswith("Q") else item or "https://www.wikidata.org",
                    title=bucket["label"],
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
        self.client = client or httpx.Client(timeout=3.5)
        self.limiter = RateLimiter(min_interval)
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 3) -> list[Evidence]:
        if not wants_newspapers(query):
            return []
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
        nara_api_key: str = "",
        smithsonian_api_key: str = "",
        dpla_api_key: str = "",
        europeana_api_key: str = "",
    ):
        from factcheck.archives import (
            CrossrefRetriever,
            DplaRetriever,
            EuropeanaRetriever,
            GdeltRetriever,
            InternetArchiveRetriever,
            NaraRetriever,
            SmithsonianRetriever,
            WikipediaRetriever,
        )

        self.wikidata = WikidataRetriever(email, client=client, min_interval=0.2)
        self.loc = ChroniclingAmericaRetriever(email, client=client, min_interval=1.0)
        self.wikipedia = WikipediaRetriever(email, client=client, min_interval=0.2)
        self.crossref = CrossrefRetriever(email, client=client, min_interval=0.4)
        self.archive = InternetArchiveRetriever(email, client=client, min_interval=0.5)
        self.gdelt = GdeltRetriever(client=client, min_interval=5.0)
        self.nara = NaraRetriever(nara_api_key, client=client) if nara_api_key else None
        self.smithsonian = SmithsonianRetriever(smithsonian_api_key, client=client) if smithsonian_api_key else None
        self.dpla = DplaRetriever(dpla_api_key, client=client) if dpla_api_key else None
        self.europeana = EuropeanaRetriever(europeana_api_key, client=client) if europeana_api_key else None
        self.google = google
        self.timeout = timeout
        self.warnings: list[str] = []

    def providers_for(self, categories: list[str] | None, query: str) -> list:
        cats = set(categories or [])
        text = query or ""
        chosen = []
        settled = well_documented_fact(text)
        if wants_reference_sources(cats, text):
            chosen.append(self.wikidata)
            chosen.append(self.wikipedia)
            if not settled:
                if wants_newspapers(text):
                    chosen.append(self.loc)
                chosen.append(self.archive)
                for optional in (self.nara, self.smithsonian, self.dpla, self.europeana):
                    if optional is not None:
                        chosen.append(optional)
            if self.google is not None:
                chosen.append(self.google)
        if cats & {"T2", "T3", "T4"} or _DOI_IN.search(text) or _SCIENCE.search(text):
            chosen.append(self.crossref)
        if _NEWS.search(text) and not settled:
            chosen.append(self.gdelt)
        if _HTTP_IN.search(text) and self.archive not in chosen:
            chosen.append(self.archive)
        return chosen

    def search(self, query: str, *, limit: int = 3, categories: list[str] | None = None) -> list[Evidence]:
        providers = self.providers_for(categories, query)
        if not providers:
            return []
        calls = [(provider, query, limit) for provider in providers]
        subject = accusation_subject(query)
        if subject and self.wikipedia in providers:
            # "X criminal record" finds pages about crime, not X. Fetch X's own biography too,
            # so a judge can see whether the authoritative record mentions any charge.
            calls.append((self.wikipedia, subject, 1))
        hits: list[Evidence] = []
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = {pool.submit(provider.search, text, limit=size): provider for provider, text, size in calls}
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
