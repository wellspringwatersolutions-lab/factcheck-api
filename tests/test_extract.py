from factcheck.extract import LiveExtractor, heuristic_extract, parse_decomposition, parse_disambiguation, parse_selection
from factcheck.llm import ScriptedLLM
from factcheck.types import UsageMeter


def test_parsers_keep_opinions_and_attribution():
    labels = parse_selection(
        {"sentences": [
            {"id": 0, "label": "fact"},
            {"id": 1, "label": "opinion"},
            {"id": 2, "label": "drop"},
            {"id": 3, "label": "nope"},
        ]},
        4,
    )
    assert labels[1] == "opinion"
    assert labels[2] == "drop"
    assert labels[3] == "drop"
    rows = parse_disambiguation(
        {"sentences": [
            {"id": 0, "text": "In 2020 the WHO said malaria deaths fell.", "drop": False, "claim_type": "fact", "attribution": "WHO"},
            {"id": 1, "text": "He won.", "drop": True, "claim_type": "fact", "attribution": None},
        ]}
    )
    assert rows[0]["attribution"] == "WHO"
    assert rows[1]["drop"] is True
    claims = parse_decomposition(
        {"claims": [
            {"text": "The WHO said malaria deaths fell in 2020.", "claim_type": "fact", "claim_category": "T4", "attribution": "WHO"},
            {"text": "The bill is a disaster.", "claim_type": "opinion", "claim_category": "T8"},
        ]}
    )
    assert claims[0].attribution == "WHO"
    assert "WHO said" in claims[0].text
    assert claims[1].claim_type == "opinion"
    assert claims[1].claim_category == "T8"


def test_live_extractor_runs_three_stages_and_falls_back():
    from factcheck.config import Settings

    scripted = ScriptedLLM([
        '{"sentences": [{"id": 0, "label": "fact"}, {"id": 1, "label": "opinion"}]}',
        '{"sentences": [{"id": 0, "text": "The WHO said malaria deaths fell in 2020.", "drop": false, "claim_type": "fact", "attribution": "WHO"}, {"id": 1, "text": "The bill is a disaster.", "drop": false, "claim_type": "opinion", "attribution": null}]}',
        '{"claims": [{"text": "The WHO said malaria deaths fell in 2020.", "claim_type": "fact", "claim_category": "T4", "attribution": "WHO"}, {"text": "The bill is a disaster.", "claim_type": "opinion", "claim_category": "T8"}]}',
    ])
    settings = Settings(_env_file=None, llm_mode="mock")
    extractor = LiveExtractor(scripted, "deepseek-ai/DeepSeek-V3.2-TEE", UsageMeter(), settings)
    claims, warnings = extractor.extract("The WHO said malaria deaths fell in 2020. The bill is a disaster.", None)
    assert warnings == []
    assert len(scripted.calls) == 3
    assert {claim.claim_type for claim in claims} == {"fact", "opinion"}

    broken = LiveExtractor(ScriptedLLM(["not json"]), "deepseek-ai/DeepSeek-V3.2-TEE", UsageMeter(), settings)
    fallback, warnings = broken.extract("The WHO said malaria deaths fell in 2020.", None)
    assert warnings
    assert fallback


def test_splitter_keeps_of_course_and_that_said():
    claims = heuristic_extract(
        "Of course. It's the Eiffel Tower in Paris. That said, the tower opened in 1889.",
        None,
    )
    blob = " ".join(claim.text for claim in claims)
    assert "Of's" not in blob
    assert "It said" not in blob
    assert any(claim.text == "It's the Eiffel Tower in Paris." for claim in claims)
    assert any(claim.text.startswith("That said") for claim in claims)
