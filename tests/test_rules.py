"""Verdicts rest on passages and tools, not on a model's vote."""

from factcheck.aggregate import decide
from factcheck.checks import content_hash
from factcheck.credibility import dedupe_syndicated
from factcheck.types import Evidence

from tests.conftest import client, make_service


def test_no_passage_is_unverified_even_if_a_judge_would_say_supported():
    decision = decide([], category="T5")
    assert decision.verdict == "Unverified"
    assert decision.supporting_quotes == []

    service, retriever, _, _ = make_service(hits=[])
    result = service.verify(
        text=None,
        claims=["The council published the 2019 tax ledger."],
        client=client(),
        charge=False,
    )
    claim = result.claims[0]
    assert claim.verdict == "Unverified"
    assert claim.sources == []
    assert service.ranker_a.calls == 0
    assert retriever.queries  # retrieval ran; it found nothing
    assert claim.confidence == claim.strength_of_support
    assert claim.confidence != 0.99


def test_one_passage_is_not_enough_to_call_it_supported():
    page = "Paris is the capital of France according to the city archive."
    evidence = [
        Evidence(
            url="https://example.org/paris",
            title="Archive",
            snippet=page,
            page_text=page,
            quote=page,
            entailment="entails",
            used=True,
            grade="D",
            source_type="secondary",
            origin_id="one",
            url_status=200,
            content_sha256=content_hash(page),
        )
    ]
    decision = decide(evidence, category="T5")
    assert decision.verdict == "Unsupported"
    assert decision.supporting_quotes == [page]
    assert decision.needs_human_review is True


def test_two_independent_passages_can_support():
    pages = {
        "https://archives.example/paris": (
            200,
            "Paris is the capital of France according to the national archive catalog.",
        ),
        "https://gazette.example/france": (
            200,
            "A gazetteer records Paris as the capital of France and a river city.",
        ),
    }
    hits = [
        Evidence(url=url, title=title, snippet=body, provider="test")
        for url, (title, body) in {
            "https://archives.example/paris": ("National archive", pages["https://archives.example/paris"][1]),
            "https://gazette.example/france": ("Gazetteer", pages["https://gazette.example/france"][1]),
        }.items()
    ]
    service, _, _, _ = make_service(hits=hits, pages=pages)
    result = service.verify(text=None, claims=["Paris is the capital of France."], client=client(), charge=False)
    claim = result.claims[0]
    assert claim.verdict == "Supported"
    assert len([source for source in claim.sources if source.used]) >= 2
    assert all(source.quote for source in claim.sources if source.used)
    assert all(source.content_sha256 for source in claim.sources)
    assert claim.confidence == claim.strength_of_support
    assert claim.confidence != 0.99


def test_fabricated_quote_is_caught():
    url = "https://news.example/mayor"
    page = "The mayor discussed the annual budget at Tuesday's meeting and took questions."
    hits = [Evidence(url=url, title="Meeting", snippet=page, provider="test")]
    service, _, _, _ = make_service(hits=hits, pages={url: (200, page)})
    result = service.verify(
        text=None,
        claims=['The mayor said "the budget is infinite and free for everyone".'],
        client=client(),
        charge=False,
    )
    claim = result.claims[0]
    assert claim.verdict != "Supported"
    names = {row.name: row for row in claim.deterministic_checks}
    assert names["quoted_text"].passed is False
    assert claim.needs_human_review is True


def test_dead_url_is_caught():
    url = "https://example.com/gone"
    hits = [Evidence(url=url, title="Missing", snippet="The council raised taxes.", provider="test")]
    service, _, _, _ = make_service(hits=hits, pages={})
    result = service.verify(
        text=None,
        claims=["The council raised property taxes in 2019."],
        client=client(),
        charge=False,
    )
    claim = result.claims[0]
    assert claim.verdict == "Unverified"
    assert any(row.name == "url_resolves" and row.passed is False for row in claim.deterministic_checks)


def test_math_is_recomputed_and_does_not_search():
    service, retriever, _, _ = make_service()
    wrong = service.verify(text=None, claims=["17 * 24 = 400"], client=client(), charge=False)
    right = service.verify(text=None, claims=["17 * 24 = 408"], client=client(fingerprint="b"), charge=False)
    assert wrong.claims[0].verdict == "Contradicted"
    assert "408" in wrong.claims[0].reasoning
    assert wrong.claims[0].tool["tool"] == "calculator"
    assert right.claims[0].verdict == "Supported"
    assert retriever.queries == []
    assert wrong.claims[0].claim_category == "T1"


def test_t8_opinion_uses_mainstream_lane():
    service, retriever, _, _ = make_service()
    result = service.verify(
        text=None,
        claims=["I think the bill is a disaster for small farms."],
        client=client(),
        charge=False,
    )
    claim = result.claims[0]
    assert claim.claim_category == "T8"
    assert claim.claim_type == "opinion"
    assert claim.verdict is None
    assert claim.mainstream_check.consensus == "disputed"
    assert claim.mainstream_check.label == "What mainstream fact-checkers say"
    assert retriever.queries == []
    assert service.ranker_a.calls == 0
    assert "not a true or false" in claim.reasoning


_EIFFEL_HTML = """<html><head><script>if (not ready) { return false; }</script></head>
<body><nav>Do not copy this page without permission.</nav>
<article><p>The Eiffel Tower is in Paris.</p></article></body></html>"""

_EVEREST_HTML = """<html><body><article><p>Mount Everest rises 8,849 meters above sea level.</p></article>
<script>var note = "not a height";</script></body></html>"""


def test_eiffel_html_noise_is_supported_by_the_article_sentence():
    url = "https://www.britannica.com/place/Eiffel-Tower"
    service, _, _, _ = make_service(
        hits=[Evidence(url=url, title="Eiffel Tower", snippet=_EIFFEL_HTML, provider="test")],
        pages={url: (200, _EIFFEL_HTML)},
    )
    result = service.verify(text=None, claims=["The Eiffel Tower is in Paris."], client=client(), charge=False)
    claim = result.claims[0]
    assert claim.verdict == "Supported"
    assert claim.strength_of_support < 0.7
    assert claim.sources[0].quote
    assert "function" not in claim.sources[0].quote
    assert "<" not in claim.sources[0].quote


def test_everest_height_mismatch_is_contradicted():
    url = "https://www.britannica.com/place/Mount-Everest"
    service, _, _, _ = make_service(
        hits=[Evidence(url=url, title="Mount Everest", snippet=_EVEREST_HTML, provider="test")],
        pages={url: (200, _EVEREST_HTML)},
    )
    result = service.verify(text=None, claims=["Mount Everest is 6,500 m tall."], client=client(), charge=False)
    claim = result.claims[0]
    assert claim.verdict == "Contradicted"
    assert "8,849" in claim.sources[0].quote or "8849" in claim.sources[0].quote


def test_fabricated_judge_quote_is_rejected():
    from factcheck.types import JudgeVote

    url = "https://www.britannica.com/place/Eiffel-Tower"
    page = "The Eiffel Tower is in Paris."

    class FakeQuoteRanker:
        calls = 0

        def rank(self, claims, passages):
            self.calls += 1
            return [
                JudgeVote(
                    role="a",
                    model="mock-a/heuristic",
                    provider="fake",
                    family="mock-a",
                    verdict="rank_only",
                    reasoning="invented",
                    stances={url: "supports"},
                    quotes={url: "The tower was built on the moon in secret."},
                )
                for _ in claims
            ]

    service, _, _, _ = make_service(
        hits=[Evidence(url=url, title="Eiffel Tower", snippet=page, provider="test")],
        pages={url: (200, page)},
    )
    fake = FakeQuoteRanker()
    service.ranker_a = fake
    service.ranker_b = fake
    service.ranker_tie = None
    result = service.verify(text=None, claims=["The Eiffel Tower is in Paris."], client=client(), charge=False)
    assert result.claims[0].verdict != "Supported"


def test_syndicated_copies_count_once():
    first = Evidence(
        url="https://www.reuters.com/world/taxes",
        title="City council raises taxes",
        snippet="The council voted 7 to 2 to raise taxes after the hearing.",
        grade="A",
        source_type="secondary",
    )
    copy = Evidence(
        url="https://medium.com/repost/taxes",
        title="City council raises taxes",
        snippet="The council voted 7 to 2 to raise taxes after the hearing.",
        grade="D",
        source_type="secondary",
    )
    kept = dedupe_syndicated([copy, first])
    assert len(kept) == 1
    assert "reuters.com" in kept[0].url
    assert copy.url in kept[0].syndicated_also


def test_hypothesis_is_not_true_or_false():
    page = "Some investors expect the product to ship, while others do not."
    evidence = [
        Evidence(
            url="corpus://note",
            title="Note",
            snippet=page,
            page_text=page,
            quote=page,
            entailment="neutral",
            used=False,
            grade="C",
            source_type="secondary",
            origin_id="n",
            url_status=200,
        )
    ]
    decision = decide(evidence, category="T8", hypothesis=True)
    assert decision.verdict == "Hypothesis"
    assert decision.hypothesis_status == "untested"
    assert decision.verdict not in {"Supported", "Contradicted"}
