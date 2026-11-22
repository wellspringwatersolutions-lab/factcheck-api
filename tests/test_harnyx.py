import logging

import httpx

from factcheck.app import create_app
from factcheck.harnyx import HarnyxReader
from factcheck.retrievers import CompositeRetriever, DesearchRetriever
from factcheck.types import Evidence

from tests.conftest import make_service


def test_harnyx_is_disabled_without_a_key_and_the_key_is_not_logged(caplog):
    caplog.set_level(logging.WARNING, logger="factcheck")
    service, _, _, settings = make_service(harnyx_api_key="secret-harnyx")
    create_app(settings=settings, service=service)
    assert HarnyxReader("").read("https://example.com/story") == ""
    assert "harnyx reader enabled" in caplog.text
    assert "secret-harnyx" not in caplog.text


def test_harnyx_reads_a_page_only_after_desearch_returns_nothing():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/web/extract"):
            return httpx.Response(200, text="")
        if request.method == "POST":
            return httpx.Response(202, json={"id": "task-1", "status": "queued"})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "citations": [{"url": "https://example.com/story", "excerpts": ["The council raised taxes."]}],
                "credits_charged": "0.10",
            },
        )

    desearch = DesearchRetriever("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    bundle = CompositeRetriever([desearch])
    bundle.harnyx = HarnyxReader("present", client=httpx.Client(transport=httpx.MockTransport(handler)), wait_seconds=1)
    text = bundle.enrich(Evidence(url="https://example.com/story", title="Taxes", snippet=""))
    assert "council raised taxes" in text
