import hashlib
import logging

from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from factcheck.app import IdentityMiddleware, create_app
from factcheck.mcp_server import current_identity
from factcheck.request_ctx import note_cost

from tests.conftest import make_service

SECRET = "tester-sample-key"
OTHER = "someone-else"
DIGEST = hashlib.sha256(SECRET.encode()).hexdigest()


class _Once:
    def __init__(self, amount: float):
        self.amount = amount
        self.cost_usd = 0.0
        self.warnings: list[str] = []

    def search(self, query: str, **_ignored):
        if self.amount:
            note_cost(self.amount)
            self.cost_usd += self.amount
            self.amount = 0.0
        return []


def _app(tmp_path, **overrides):
    service, _, _, settings = make_service(data_dir=str(tmp_path), **overrides)
    return create_app(settings=settings, service=service), service


def test_a_budgeted_tester_key_is_free_until_the_lifetime_budget_is_gone(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    app, _service = _app(
        tmp_path,
        retriever=_Once(0.30),
        tester_key_budgets=f"{DIGEST}:0.50",
        free_check_reserve_usd=0.25,
    )
    claim = {"claims": ["The Eiffel Tower is located in Paris."]}
    with TestClient(app) as client:
        first = client.post("/check", json=claim, headers={"X-API-Key": SECRET})
        budget = client.get("/v1/tester/budget", headers={"Authorization": f"Bearer {SECRET}"})
        second = client.post("/v1/verify", json=claim, headers={"Authorization": f"Bearer {SECRET}"})
        missing = client.get("/v1/tester/budget")
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["usage"]["caller"] == "tester"
    assert body["usage"]["tier"] == "tester"
    assert body["usage"]["tester_budget_remaining_usd"] == 0.2
    assert budget.status_code == 200
    assert budget.json()["remaining_usd"] == 0.2
    assert budget.json()["hash"] == DIGEST[:8]
    assert second.status_code == 402
    assert second.json()["error"]["message"] == "tester budget exhausted"
    assert second.json()["error"]["remaining_usd"] == 0.2
    assert missing.status_code == 401
    assert SECRET not in caplog.text
    assert "caller=tester" in caplog.text
    assert DIGEST[:8] in caplog.text


def test_the_spend_file_and_the_env_floor_both_survive_a_new_process(tmp_path):
    app, _service = _app(tmp_path, retriever=_Once(0.30), tester_key_budgets=f"{DIGEST}:0.50", free_check_reserve_usd=0.25)
    with TestClient(app) as client:
        assert client.post("/check", json={"text": "2 + 2 = 4"}, headers={"X-API-Key": SECRET}).status_code == 200
    again, _service = _app(
        tmp_path,
        tester_key_budgets=f"{DIGEST}:0.50",
        tester_key_spent=f"{DIGEST[:8]}:0.45",
        free_check_reserve_usd=0.25,
    )
    with TestClient(again) as client:
        denied = client.post("/check", json={"text": "2 + 2 = 4"}, headers={"X-API-Key": SECRET})
    assert denied.status_code == 402
    assert denied.json()["error"]["message"] == "tester budget exhausted"


def test_sha256_list_is_unlimited_but_a_budget_entry_wins_and_the_daily_cap_still_applies(tmp_path):
    unlimited, _service = _app(tmp_path / "free", tester_key_sha256s=DIGEST)
    with TestClient(unlimited) as client:
        first = client.post("/v1/verify", json={"claims": ["2 + 2 = 4"]}, headers={"X-API-Key": SECRET})
        second = client.post("/check", json={"text": "2 + 2 = 4"}, headers={"X-API-Key": SECRET})
    assert first.status_code == 200
    assert first.json()["usage"]["tester_budget_remaining_usd"] is None
    assert second.status_code == 200

    preferred, _service = _app(
        tmp_path / "budget",
        tester_key_sha256s=DIGEST,
        tester_key_budgets=f"{DIGEST}:0.10",
        free_check_reserve_usd=0.25,
    )
    with TestClient(preferred) as client:
        denied = client.post("/check", json={"text": "2 + 2 = 4"}, headers={"X-API-Key": SECRET})
    assert denied.status_code == 402

    capped, _service = _app(
        tmp_path / "cap",
        retriever=_Once(0.5),
        tester_key_sha256s=DIGEST,
        free_daily_spend_cap_usd=0.6,
        free_check_reserve_usd=0.5,
    )
    claim = {"claims": ["The Eiffel Tower is located in Paris."]}
    with TestClient(capped) as client:
        ok = client.post("/v1/verify", json=claim, headers={"X-API-Key": SECRET})
        blocked = client.post("/v1/verify", json=claim, headers={"X-API-Key": SECRET})
    assert ok.status_code == 200, ok.text
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "spend_cap_exceeded"


def test_a_wrong_key_is_not_a_tester_and_the_owner_key_still_works_as_bearer(tmp_path):
    app, _service = _app(tmp_path, tester_key_budgets=f"{DIGEST}:1", owner_key="owner-sample")
    with TestClient(app) as client:
        stranger = client.post("/check", json={"text": "2 + 2 = 4"}, headers={"X-API-Key": OTHER})
        owner = client.post("/check", json={"text": "2 + 2 = 4"}, headers={"Authorization": "Bearer owner-sample"})
        docs = client.get("/llms.txt").text
        schema = client.get("/openapi.json").json()
    assert stranger.status_code == 401
    assert owner.status_code == 200
    assert owner.json()["usage"]["tier"] == "owner"
    assert "TESTER_KEY_BUDGETS" in docs
    assert SECRET not in docs
    assert "tester key" in schema["info"]["description"].lower()
    assert SECRET not in str(schema["components"]["securitySchemes"])


def test_mcp_requests_see_the_tester_key(tmp_path):
    _app_unused, service = _app(tmp_path, tester_key_budgets=f"{DIGEST}:1")

    async def probe(request):
        ident = current_identity()
        return JSONResponse({"tester": ident.tester, "hash": ident.tester_hash[:8]})

    app = Starlette(routes=[Route("/mcp", probe)])
    app.add_middleware(IdentityMiddleware, settings=service.settings, service=service)
    with TestClient(app) as client:
        seen = client.get("/mcp", headers={"X-API-Key": SECRET})
    assert seen.json() == {"tester": True, "hash": DIGEST[:8]}
