from factcheck.mainstream import build_result, consensus, from_google_payload, polarity
from factcheck.types import OutletCheck


def test_polarity_and_consensus():
    assert polarity("Pants on Fire") == "disputes"
    assert polarity("Mostly True") == "agrees"
    assert polarity("Half True") == "mixed"
    checks = [
        OutletCheck("Snopes", "False", "https://snopes.com/a", "2024-01-01", "disputes"),
        OutletCheck("PolitiFact", "Pants on Fire", "https://politifact.com/a", "2024-02-01", "disputes"),
    ]
    assert consensus(checks) == "disputed"
    mixed = checks + [OutletCheck("FactCheck.org", "True", "https://factcheck.org/a", None, "agrees")]
    assert consensus(mixed) == "mixed"
    assert consensus([]) == "no_coverage"
    result = build_result(checks)
    assert result.consensus == "disputed"
    assert "not a true or false" in result.reasoning


def test_google_payload_skips_unrelated_claims():
    payload = {
        "claims": [
            {
                "text": "The moon is made of cheese",
                "claimReview": [
                    {"publisher": {"name": "Snopes"}, "url": "https://www.snopes.com/fact-check/cheese", "textualRating": "False", "reviewDate": "2020-01-01"}
                ],
            },
            {
                "text": "The city council raised taxes",
                "claimReview": [
                    {"publisher": {"name": "PolitiFact"}, "url": "https://www.politifact.com/article/taxes", "textualRating": "Mostly False", "reviewDate": "2024-03-01"}
                ],
            },
        ]
    }
    found = from_google_payload(payload, "The city council raised taxes in the last budget")
    assert len(found) == 1
    assert found[0].outlet == "PolitiFact"
    assert found[0].polarity == "disputes"
