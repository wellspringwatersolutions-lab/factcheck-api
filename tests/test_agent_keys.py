import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from factcheck.app import IdentityMiddleware, create_app
from factcheck.config import Settings
from factcheck.mcp_server import current_identity

from tests.conftest import make_service
from tests.test_tester import _Once

SECRET = "agent-sample-key"
DIGEST = hashlib.sha256(SECRET.encode()).hexdigest()


def test_no_agent_keys_are_configured_by_default():
    assert Settings(_env_file=None, database_url="sqlite://", llm_mode="mock").agent_key_daily == ""


def test_an_agent_key_resets_daily_and_stops_when_the_day_is_spent(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    entry = f"desk:{DIGEST}:0.50"
    service, _, _, settings = make_service(
        data_dir=str(tmp_path),
        retriever=_Once(0.30),
        agent_key_daily=entry,
        free_check_reserve_usd=0.25,
    )
    app = create_app(settings=settings, service=service)
    claim = {"claims": ["The Eiffel Tower is located in Paris."]}
    with TestClient(app) as client:
        first = client.post("/check", json=claim, headers={"X-API-Key": SECRET})
        second = client.post("/v1/verify", json=claim, headers={"Authorization": f"Bearer {SECRET}"})
    assert first.status_code == 200, first.text
    assert first.json()["usage"]["caller"] == "desk"
    assert first.json()["usage"]["budget_remaining_usd"] == 0.2
    assert second.status_code == 402
    assert second.json()["error"]["message"] == "agent daily budget exhausted"
    assert second.json()["error"]["remaining_usd"] == 0.2
    assert second.json()["error"]["reset_at"].endswith("+00:00")
    assert SECRET not in caplog.text
    assert "caller=desk" in caplog.text

    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    (tmp_path / "agent_spend.json").write_text(json.dumps({"day": yesterday, "spent": {DIGEST: 0.50}}))
    fresh, _, _, fresh_settings = make_service(data_dir=str(tmp_path), agent_key_daily=entry, free_check_reserve_usd=0.25)
    with TestClient(create_app(settings=fresh_settings, service=fresh)) as client:
        rolled = client.post("/check", json={"text": "2 + 2 = 4"}, headers={"X-API-Key": SECRET})
    assert rolled.status_code == 200, rolled.text
    assert rolled.json()["usage"]["budget_remaining_usd"] == 0.5


def test_mcp_requests_see_an_agent_key(tmp_path):
    service, _, _, settings = make_service(data_dir=str(tmp_path), agent_key_daily=f"desk:{DIGEST}:1")

    async def probe(request):
        ident = current_identity()
        return JSONResponse({"name": ident.agent_name, "hash": ident.agent_hash[:8]})

    app = Starlette(routes=[Route("/mcp", probe)])
    app.add_middleware(IdentityMiddleware, settings=settings, service=service)
    with TestClient(app) as client:
        seen = client.get("/mcp", headers={"Authorization": f"Bearer {SECRET}"})
    assert seen.json() == {"name": "desk", "hash": DIGEST[:8]}
