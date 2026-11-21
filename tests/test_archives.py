import httpx

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
from factcheck.references import ReferenceBundle


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_wikipedia_search_and_extract_share_one_origin():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "factcheck/0.1" in request.headers["user-agent"]
        assert "mailto:lab@example.com" in request.headers["user-agent"]
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [{"title": "Ada Lovelace", "snippet": "Born 1815."}]}})
        assert request.url.params.get("prop") == "extracts"
        return httpx.Response(
            200,
            json={"query": {"pages": [{"title": "Ada Lovelace", "extract": "Ada Lovelace was born in 1815."}]}},
        )

    hits = WikipediaRetriever("lab@example.com", client=_client(handler), min_interval=0).search("Ada Lovelace")
    assert hits[0].url.endswith("Ada_Lovelace")
    assert hits[0].source_type == "secondary"
    assert hits[0].origin_id == "wikipedia.org"
    assert "1815" in hits[0].page_text


def test_crossref_parses_a_work_and_sends_mailto():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["mailto"] == "lab@example.com"
        assert "10.1038/nature" in str(request.url)
        return httpx.Response(
            200,
            json={"message": {"DOI": "10.1038/nature12373", "title": ["A paper"], "issued": {"date-parts": [[2013]]}}},
        )

    hits = CrossrefRetriever("lab@example.com", client=_client(handler), min_interval=0).search("doi 10.1038/nature12373")
    assert hits[0].url == "https://doi.org/10.1038/nature12373"
    assert hits[0].source_type == "peer_reviewed"
    assert hits[0].published_at == "2013"


def test_wayback_and_archive_search():
    def wayback(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "archive.org"
        return httpx.Response(
            200,
            json={"archived_snapshots": {"closest": {"available": True, "url": "https://web.archive.org/web/20200101/https://example.com/", "timestamp": "20200101000000"}}},
        )

    snap = InternetArchiveRetriever("lab@example.com", client=_client(wayback), min_interval=0).search("see https://example.com/old")
    assert "web.archive.org" in snap[0].url
    assert snap[0].study_type == "archive_snapshot"

    def search(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/advancedsearch.php")
        return httpx.Response(200, json={"response": {"docs": [{"identifier": "lincolnpapers", "title": "Lincoln papers", "description": "Letters from 1865.", "date": "1865"}]}})

    docs = InternetArchiveRetriever("lab@example.com", client=_client(search), min_interval=0).search("Lincoln papers 1865")
    assert docs[0].url.endswith("/lincolnpapers")
    assert docs[0].source_type == "primary_source"


def test_gdelt_parses_articles():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["mode"] == "ArtList"
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json={"articles": [{"url": "https://news.example/a", "title": "Election results", "seendate": "20241101T000000Z"}]})

    hits = GdeltRetriever(client=_client(handler), min_interval=0).search("election 2024")
    assert hits[0].title == "Election results"
    assert hits[0].source_type == "secondary"


def test_keyed_catalogs_skip_when_blank_and_parse_when_set():
    assert NaraRetriever("").search("Lincoln") == []
    assert SmithsonianRetriever("").search("Lincoln") == []
    assert DplaRetriever("").search("Lincoln") == []
    assert EuropeanaRetriever("").search("Lincoln") == []

    def nara(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "nara-key"
        return httpx.Response(200, json={"result": [{"naId": "99", "title": "Lincoln letter", "scopeAndContentNote": "Written in 1865."}]})

    assert "1865" in NaraRetriever("nara-key", client=_client(nara), min_interval=0).search("Lincoln")[0].page_text

    def si(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "si-key"
        return httpx.Response(
            200,
            json={"response": {"rows": [{"content": {"descriptiveNonRepeating": {"title": {"content": "Portrait"}, "record_link": "https://si.edu/object/1"}, "freetext": {"notes": [{"content": "Oil on canvas."}]}}}]}},
        )

    assert SmithsonianRetriever("si-key", client=_client(si), min_interval=0).search("portrait")[0].title == "Portrait"

    def dpla(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "dpla-key"
        return httpx.Response(200, json={"docs": [{"id": "abc", "isShownAt": "https://library.example/item", "sourceResource": {"title": "Broadside", "description": "Printed in 1865."}}]})

    assert "1865" in DplaRetriever("dpla-key", client=_client(dpla), min_interval=0).search("broadside")[0].snippet

    def europeana(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "eu-key"
        return httpx.Response(200, json={"items": [{"title": ["Letter"], "dcDescription": ["From 1865."], "guid": "https://www.europeana.eu/item/1"}]})

    assert EuropeanaRetriever("eu-key", client=_client(europeana), min_interval=0).search("letter")[0].url.endswith("/1")


def test_routing_history_news_and_keys():
    bare = ReferenceBundle(email="lab@example.com")
    history = [item.name for item in bare.providers_for(["T5"], "Lincoln was shot in 1865")]
    assert "wikipedia" in history and "internet_archive" in history and "nara" not in history
    assert bare.providers_for(["T4"], "a randomized vaccine trial") == [bare.crossref]
    news = [item.name for item in bare.providers_for(["T6"], "the election today")]
    assert news == ["gdelt"]
    keyed = ReferenceBundle(email="lab@example.com", nara_api_key="n", smithsonian_api_key="s", dpla_api_key="d", europeana_api_key="e")
    names = [item.name for item in keyed.providers_for(["T5"], "Lincoln 1865")]
    assert {"nara", "smithsonian", "dpla", "europeana"} <= set(names)
