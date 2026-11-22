"""Public verdicts are five names. Internal likely_true still counts toward the rating."""

from factcheck.publish import canonical_verdict, map_verdict, publish_verification
from factcheck.schemas import ClaimOut, SynopsisOut, VerificationOut
from factcheck.synopsis import _phrasing_ok, build_synopsis


def _claim(text, verdict, strength=0.6):
    return ClaimOut(
        text=text,
        claim_type="fact",
        claim_category="T6",
        verdict=verdict,
        strength_of_support=strength,
        confidence=strength,
        reasoning="Checked.",
    )


def test_public_mapping_and_old_names_on_input():
    assert map_verdict("likely_true", "fact") == ("hypothesis", "likely_true", "Hypothesis")
    assert map_verdict("Hypothesis", "fact") == ("hypothesis", "speculative", "Hypothesis")
    assert map_verdict("Contradicted", "fact") == ("conflicting", "contradicted", "Conflicting")
    assert map_verdict("Unsupported", "fact") == ("conflicting", None, "Conflicting")
    assert map_verdict("Supported", "fact") == ("supported", None, "Supported")
    assert map_verdict(None, "opinion") == ("opinion", None, "Opinion")
    assert map_verdict("hypothesis", "fact", "likely_true") == ("hypothesis", "likely_true", "Hypothesis")
    assert canonical_verdict("supported") == "Supported"
    assert canonical_verdict("likely_true") == "likely_true"
    assert canonical_verdict("contradicted") == "Contradicted"
    assert canonical_verdict("conflicting") == "Conflicting"


def test_likely_true_still_rates_and_publishes_as_hypothesis():
    denial = "The U.S. government has not told civilians to hoard 90 days of food."
    rows = [
        _claim(denial, "likely_true", 0.6),
        _claim("One local detail could not be pinned down.", "Unverified", 0.1),
    ]
    synopsis = build_synopsis(rows)
    assert synopsis.rating == "Partly verified"
    result = VerificationOut(
        verification_id="abc",
        created_at="2026-11-22T00:00:00+00:00",
        input_summary="paste",
        claims=rows,
        summary="",
        attribution="",
        disclaimer="",
        synopsis=synopsis,
        usage={"tier": "free"},
        upgrade_url="",
    )
    published = publish_verification(result)
    claim = published["claims"][0]
    assert claim["verdict"] == "hypothesis"
    assert claim["verdict_detail"] == "likely_true"
    assert claim["verdict_label"] == "Hypothesis"
    assert published["claims"][1]["verdict"] == "unverified"
    assert set(item["verdict"] for item in published["claims"]) <= {
        "supported",
        "hypothesis",
        "opinion",
        "conflicting",
        "unverified",
    }


def test_chutes_hedges_likely_true_and_marks_a_contradiction_as_disputed():
    denial = "The U.S. government has not told civilians to hoard 90 days of food."
    rows = [
        _claim(denial, "likely_true", 0.6),
        _claim("One local detail could not be pinned down.", "Unverified", 0.1),
    ]
    flat = "Partly verified. The U.S. government has not told civilians to hoard 90 days of food. One claim could not be confirmed."
    assert not _phrasing_ok(flat, denial, "Partly verified", rows)
    hedged_tldr = "Reputable sources report that the U.S. government has not told civilians to hoard 90 days of food."
    hedged = f"Partly verified. {hedged_tldr} One claim could not be confirmed."
    assert _phrasing_ok(hedged, hedged_tldr, "Partly verified", rows)

    disputed = [
        _claim("Mount Everest is 6500 m tall.", "Contradicted", 0.8),
        _claim("Mount Everest is 8849 m tall.", "Supported", 0.8),
    ]
    bad = "Mixed. Mount Everest is 6500 m tall. The surveyed height is 8849 m."
    assert not _phrasing_ok(bad, "Mount Everest is 6500 m tall.", "Mixed", disputed)
    good_tldr = "Sources contradict the claim that Mount Everest is 6500 m tall."
    good = f"Mixed. {good_tldr} The surveyed height is 8849 m."
    assert _phrasing_ok(good, good_tldr, "Mixed", disputed)
