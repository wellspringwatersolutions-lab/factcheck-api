import httpx

from factcheck.checks import UrlChecker, passage_check_rows
from factcheck.references import ReferenceBundle, WikidataRetriever
from factcheck.types import Evidence

from tests.conftest import client, make_service

_WIKIDATA = (
    "Wikidata: George Washington. "
    "Position held: President of the United States (ordinal 1). "
    "Position held: Commander in Chief of the Continental Army."
)
_NPS = (
    "The Second Continental Congress commissioned George Washington as "
    "Commander in Chief of the Continental Army on June 19, 1775."
)
_WIKI = "https://en.wikipedia.org/wiki/George_Washington"
_NPS_URL = "https://www.nps.gov/articles/george-washington.htm"


def test_blocked_wikipedia_nps_snippet_and_wikidata_office_support_the_claims():
    hits = [
        Evidence(
            url=_WIKI,
            title="George Washington",
            snippet="George Washington was the first president of the United States.",
            provider="desearch",
        ),
        Evidence(url=_NPS_URL, title="National Park Service", snippet=_NPS, page_text=_NPS, provider="desearch", grade="A"),
        Evidence(
            url="https://www.wikidata.org/wiki/Q23",
            title="George Washington",
            snippet=_WIKIDATA,
            page_text=_WIKIDATA,
            provider="wikidata",
            source_type="structured",
            grade="A",
        ),
    ]
    pages = {_WIKI: (403, "Forbidden"), _NPS_URL: (200, _NPS)}
    service, _, _, _ = make_service(hits=hits, pages=pages)
    president = service.verify(
        text=None,
        claims=["George Washington was the first President of the United States."],
        client=client(),
        charge=False,
    ).claims[0]
    general = service.verify(
        text=None,
        claims=["George Washington was commanding general of the Continental Army."],
        client=client(fingerprint="gw-2"),
        charge=False,
    ).claims[0]
    assert president.verdict == "Supported"
    assert general.verdict == "Supported"
    assert any("ordinal 1" in source.quote for source in president.sources)
    assert any("Commander in Chief" in (source.quote or "") for source in general.sources)
    assert president.verdict != "Unsupported"
    failed = [row for row in president.deterministic_checks if row.name == "url_resolves" and not row.passed]
    assert failed == []
    assert not any(row.detail.startswith(_WIKI) and row.passed is False for row in passage_check_rows(president.text, []))


def test_wikidata_parses_position_held():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "P39" in request.url.params["query"]
        return httpx.Response(
            200,
            json={
                "results": {
                    "bindings": [
                        {
                            "item": {"value": "http://www.wikidata.org/entity/Q23"},
                            "itemLabel": {"value": "George Washington"},
                            "officeLabel": {"value": "President of the United States"},
                            "ordinal": {"value": "1"},
                        }
                    ]
                }
            },
        )

    hits = WikidataRetriever("lab@example.com", client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0).search(
        "George Washington"
    )
    assert "Position held: President of the United States (ordinal 1)." in hits[0].page_text


def test_loc_skips_office_facts_and_post_1963():
    bundle = ReferenceBundle(email="lab@example.com")
    office = [item.name for item in bundle.providers_for(["T5"], "George Washington was the first President of the United States")]
    later = [item.name for item in bundle.providers_for(["T5"], "A newspaper said so in 1980")]
    lincoln = [item.name for item in bundle.providers_for(["T5"], "Lincoln was shot in 1865")]
    assert "loc" not in office
    assert "loc" not in later
    assert "loc" in lincoln


def test_fetcher_sends_a_user_agent_uses_wikipedia_api_and_keeps_a_snippet_after_403():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        assert request.headers["user-agent"].startswith("factcheck/0.1")
        assert "mailto:lab@example.com" in request.headers["user-agent"]
        if request.url.host == "en.wikipedia.org":
            assert request.url.path == "/w/api.php"
            return httpx.Response(200, json={"query": {"pages": [{"extract": "George Washington was the first president of the United States."}]}})
        assert request.url.host == "www.britannica.com"
        return httpx.Response(403, text="Forbidden")

    checker = UrlChecker(client=httpx.Client(transport=httpx.MockTransport(handler)), email="lab@example.com")
    wiki = checker.resolve("https://en.wikipedia.org/wiki/George_Washington")
    assert wiki.ok and wiki.status == 200
    assert "first president" in wiki.body
    blocked = checker.resolve(
        "https://www.britannica.com/biography/George-Washington",
        body="George Washington was the first president of the United States.",
    )
    assert blocked.ok and blocked.status == 200
    assert "first president" in blocked.body
    assert "en.wikipedia.org" in seen and "www.britannica.com" in seen
