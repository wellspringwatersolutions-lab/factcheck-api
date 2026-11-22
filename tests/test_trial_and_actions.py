from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from factcheck.app import IdentityMiddleware, create_app
from factcheck.mcp_server import current_identity

from tests.conftest import make_service


def _apps(tmp_path):
    service, _, _, settings = make_service(data_dir=str(tmp_path))
    return create_app(settings=settings, service=service), service


def test_a_trial_key_allows_three_checks_and_one_key_per_ip(tmp_path, caplog):
    app, service = _apps(tmp_path)
    with TestClient(app) as client:
        issued = client.post("/v1/keys/trial", headers={"X-Forwarded-For": "198.51.100.20"})
        assert issued.status_code == 200, issued.text
        raw = issued.json()["api_key"]
        assert issued.json()["checks_remaining"] == 3
        again = client.post("/v1/keys/trial", headers={"X-Forwarded-For": "198.51.100.20"})
        other = client.post("/v1/keys/trial", headers={"X-Forwarded-For": "198.51.100.21"})
        statuses = []
        for _ in range(4):
            statuses.append(client.post("/check", json={"text": "2 + 2 = 4"}, headers={"Authorization": f"Bearer {raw}"}).status_code)
    assert again.status_code == 429
    assert other.status_code == 200
    assert statuses == [200, 200, 200, 402]
    stored = (tmp_path / "trial_keys.json").read_text()
    assert raw not in stored
    assert raw not in caplog.text

    restarted, restarted_service = _apps(tmp_path)

    async def probe(request):
        ident = current_identity()
        return JSONResponse({"trial": ident.trial})

    probe_app = Starlette(routes=[Route("/mcp", probe)])
    probe_app.add_middleware(IdentityMiddleware, settings=restarted_service.settings, service=restarted_service)
    with TestClient(probe_app) as client:
        seen = client.get("/mcp", headers={"X-API-Key": raw})
    assert seen.json()["trial"] is True
    with TestClient(restarted) as client:
        kept = client.post("/v1/verify", json={"claims": ["2 + 2 = 4"]}, headers={"X-API-Key": other.json()["api_key"]})
    assert kept.status_code == 200


def test_every_verdict_has_a_confidence_and_a_recommended_action(tmp_path):
    app, _service = _apps(tmp_path)
    with TestClient(app) as client:
        math = client.post("/v1/verify", json={"claims": ["17 * 24 = 408"]}, headers={"X-Client-Fingerprint": "act-1"})
        opinion = client.post("/check", json={"text": "I think the bill is a disaster for small farms."}, headers={"X-Client-Fingerprint": "act-2"})
        schema = client.get("/openapi.json").text
        llms = client.get("/llms.txt").text
        card = client.get("/.well-known/mcp.json").json()
    assert math.status_code == 200, math.text
    claim = math.json()["claims"][0]
    assert 0 <= claim["confidence"] <= 1
    assert claim["recommended_action"] == "safe_to_state"
    assert opinion.status_code == 200, opinion.text
    assert opinion.json()["claims"][0]["recommended_action"] == "do_not_state"
    assert "recommended_action" in schema
    assert "safe_to_state" in schema
    assert "Claude Desktop" in llms
    assert "POST" in card["connect"]["streamable_http"]
    assert any(tool["name"] == "report_bad_verdict" for tool in card["tools"])
