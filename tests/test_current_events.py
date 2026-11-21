import httpx

from factcheck.aggregate import decide
from factcheck.checks import best_passages
from factcheck.cleanup import entity_query, prepare_answer
from factcheck.config import Settings
from factcheck.credibility import grade_for
from factcheck.extract import LiveExtractor
from factcheck.llm import ScriptedLLM
from factcheck.orchestrator import request_budget
from factcheck.retrievers import DesearchRetriever
from factcheck.types import Evidence, UsageMeter

_QUOTE_A = "French supermarket refrigerators were knocked offline for two days after the intrusion."
_QUOTE_B = "Investigators said the refrigeration controllers at the French grocer were remotely disabled."


def _news(url: str, quote: str) -> Evidence:
    return Evidence(
        url=url,
        title="Refrigeration hack",
        snippet=quote,
        page_text=quote,
        quote=quote,
        entailment="entails",
        used=True,
        grade=grade_for(url),
        source_type="secondary",
        origin_id=url,
    )


def test_major_outlets_are_b_and_two_of_them_support_a_current_event():
    assert grade_for("https://www.cbsnews.com/hack") == "B"
    assert grade_for("https://www.aljazeera.com/hack") == "B"
    assert grade_for("https://www.newsweek.com/hack") == "B"
    assert grade_for("https://thehackernews.com/hack") == "C"
    assert grade_for("https://www.csis.org/hack") == "C"
    assert grade_for("https://www.delawareonline.com/hack") == "C"
    assert grade_for("https://www.msb.se/en") == "A"
    assert grade_for("https://europa.eu/preparedness") == "A"
    assert grade_for("https://www.snopes.com/fact-check/hack") == "B"
    assert grade_for("https://random-shop.example/blog") == "E"
    decision = decide(
        [
            _news("https://www.cbsnews.com/french-hack", _QUOTE_A),
            _news("https://www.aljazeera.com/french-hack", _QUOTE_B),
        ],
        category="T6",
        claim_text="A French supermarket refrigeration system was hacked in 2024.",
    )
    assert decision.verdict == "Supported"
    titled = _news("https://www.cbsnews.com/french-hack", _QUOTE_A)
    titled.quote = titled.title
    thin = decide(
        [titled, _news("https://www.aljazeera.com/french-hack", "Published: 2024-11-01")],
        category="T6",
        claim_text="A French supermarket refrigeration system was hacked in 2024.",
    )
    assert thin.verdict != "Supported"


def test_best_passages_skip_the_title_and_the_dateline():
    page = "\n".join(
        [
            "French grocer hacked",
            "Published: November 1, 2024",
            "Share this story with friends.",
            "French supermarket refrigerators were knocked offline for two days after the intrusion.",
            "A later paragraph discusses an unrelated sports score from the same city.",
        ]
    )
    chosen = best_passages(page, "French supermarket refrigerators were hacked", title="French grocer hacked")
    assert chosen
    assert chosen[0].startswith("French supermarket refrigerators")
    assert all(not item.lower().startswith("published") for item in chosen)
    assert all(item != "French grocer hacked" for item in chosen)


def test_extraction_drops_headings_and_repairs_json():
    pasted = "\n".join(
        [
            "The Real Goal: Buying Governments Time",
            "The situation breaks down into three parts.",
            "The United States (A misunderstood military order): The Pentagon gave commanders 90 days.",
            "Stores ran out of bottled water after the warning. CBC+ 2",
        ]
    )
    prepared = prepare_answer(pasted)
    assert "Real Goal" not in prepared
    assert "breaks down" not in prepared
    assert "misunderstood military order" not in prepared
    assert "Pentagon gave commanders 90 days" in prepared
    assert "CBC" not in prepared
    assert entity_query("The FORTRESS America order gives Hegseth 90 days to harden bases after the hack.") == (
        "The FORTRESS America Hegseth 90 days bases hack"
    )
    settings = Settings(_env_file=None, llm_mode="mock")
    extractor = LiveExtractor(
        ScriptedLLM(["{not json", '{"sentences": [{"id": 0, "label": "fact",}],}']),
        "deepseek-ai/DeepSeek-V3.2-TEE",
        UsageMeter(),
        settings,
    )
    payload = extractor._ask("Return JSON", "[0] The Pentagon gave commanders 90 days.")
    assert payload["sentences"][0]["label"] == "fact"


def test_recent_claims_use_desearch_news_and_long_text_gets_75_seconds():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/news"
        assert request.url.params["time"] == "week"
        assert "FORTRESS" in request.url.params["query"]
        assert "gives Hegseth" not in request.url.params["query"]
        return httpx.Response(200, json={"data": []}, headers={"X-Desearch-Cost-Usd": "0"})

    DesearchRetriever("k", client=httpx.Client(transport=httpx.MockTransport(handler))).search(
        "The FORTRESS America order gives Hegseth 90 days to harden bases after the hack."
    )
    settings = Settings(_env_file=None, llm_mode="mock")
    assert settings.max_checked_claims == 10
    assert settings.check_concurrency == 6
    assert request_budget(settings, "short") == 45
    assert request_budget(settings, "x" * 4000) == 75
