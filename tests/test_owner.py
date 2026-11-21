import logging

from fastapi.testclient import TestClient

from factcheck.app import create_app
from factcheck.billing import owner_key_matches, resolve_client_ip

from tests.conftest import make_service


def test_client_ip_uses_the_first_forwarded_hop():
    assert resolve_client_ip({"x-forwarded-for": "198.51.100.8, 10.0.0.4"}, "10.0.0.1") == "198.51.100.8"
    assert resolve_client_ip({}, "10.0.0.1") == "10.0.0.1"


def test_owner_key_compare_is_exact_and_empty_never_matches():
    assert owner_key_matches("correct-horse", "correct-horse")
    assert not owner_key_matches("wrong", "correct-horse")
    assert not owner_key_matches("", "correct-horse")
    assert not owner_key_matches("correct-horse", "")


def test_owner_ip_skips_the_per_caller_limit(caplog):
    caplog.set_level(logging.INFO)
    service, _, _, settings = make_service(owner_ips="198.51.100.8, 203.0.113.4")
    app = create_app(settings=settings, service=service)
    headers = {"X-Forwarded-For": "203.0.113.4, 10.0.0.9", "X-Client-Fingerprint": "owner-browser"}
    with TestClient(app) as client:
        first = client.post("/check", json={"text": "2 + 2 = 4"}, headers=headers)
        second = client.post("/v1/verify", json={"claims": ["2 + 2 = 4"]}, headers=headers)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["usage"]["tier"] == "owner"
    assert "tier=owner" in caplog.text
    assert "203.0.113.4" in caplog.text


def test_owner_key_allows_repeat_checks_and_a_wrong_key_does_not(caplog):
    caplog.set_level(logging.INFO)
    service, _, _, settings = make_service(owner_key="correct-horse")
    app = create_app(settings=settings, service=service)
    with TestClient(app) as client:
        ok = client.post(
            "/v1/verify",
            json={"claims": ["2 + 2 = 4"]},
            headers={"X-Owner-Key": "correct-horse", "X-Client-Fingerprint": "key-user"},
        )
        again = client.post(
            "/check",
            json={"text": "2 + 2 = 4"},
            headers={"X-Owner-Key": "correct-horse", "X-Client-Fingerprint": "key-user"},
        )
        wrong = client.post(
            "/v1/verify",
            json={"claims": ["2 + 2 = 4"]},
            headers={"X-Owner-Key": "nope", "X-Client-Fingerprint": "stranger"},
        )
        blocked = client.post(
            "/v1/verify",
            json={"claims": ["2 + 2 = 4"]},
            headers={"X-Owner-Key": "nope", "X-Client-Fingerprint": "stranger"},
        )
    assert ok.status_code == 200
    assert ok.json()["usage"]["tier"] == "owner"
    assert again.status_code == 200
    assert wrong.status_code == 200
    assert wrong.json()["usage"]["tier"] == "free"
    assert blocked.status_code == 402
    assert blocked.json()["error"]["code"] == "free_tier_exhausted"
    assert "correct-horse" not in caplog.text
    assert "nope" not in caplog.text


class _CostRetriever:
    def __init__(self):
        self.cost_usd = 0.0
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 6, scientific: bool = False, categories=None, **_ignored):
        from factcheck.request_ctx import note_cost

        self.cost_usd += 0.5
        note_cost(0.5)
        return []


def test_owner_still_hits_the_daily_spend_cap():
    service, _, _, settings = make_service(
        retriever=_CostRetriever(),
        owner_key="correct-horse",
        free_daily_spend_cap_usd=0.6,
        free_check_reserve_usd=0.5,
    )
    app = create_app(settings=settings, service=service)
    headers = {"X-Owner-Key": "correct-horse"}
    claim = {"claims": ["The Eiffel Tower is located in Paris."]}
    with TestClient(app) as client:
        first = client.post("/v1/verify", json=claim, headers=headers)
        second = client.post("/v1/verify", json=claim, headers=headers)
    assert first.status_code == 200, first.text
    assert first.json()["usage"]["tier"] == "owner"
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "spend_cap_exceeded"
