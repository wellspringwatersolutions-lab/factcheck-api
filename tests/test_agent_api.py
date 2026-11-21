"""Agent-facing docs, request ids, and the async MCP tools."""

import asyncio
import json
import logging

from fastapi.testclient import TestClient

from factcheck.app import create_app
from factcheck.mcp_server import build_mcp

from tests.conftest import make_service


def test_constitution_is_markdown_and_json_and_linked():
    service, _, _, settings = make_service()
    app = create_app(settings=settings, service=service)
    with TestClient(app) as client:
        markdown = client.get("/constitution")
        assert markdown.status_code == 200
        assert "text/markdown" in markdown.headers["content-type"]
        assert "likely_true" in markdown.text
        assert "Contradicted" in markdown.text
        document = client.get("/constitution", headers={"Accept": "application/json"})
        assert document.status_code == 200
        body = document.json()
        assert body["version"]
        assert body["markdown"].startswith("# Factcheck constitution")
        assert any(item["verdict"] == "likely_true" for item in body["verdicts"])
        llms = client.get("/llms.txt").text
        assert "/constitution" in llms
        assert "claims[].text" in llms or "`text`" in llms
        assert "`claim`" in llms
        assert "120" in llms
        assert "Authorization: Bearer" in llms
        assert "X-API-Key" in llms
        assert "start_check" in llms
        assert "get_job" in llms


def test_openapi_describes_both_shapes_errors_and_auth():
    service, _, _, settings = make_service()
    app = create_app(settings=settings, service=service)
    schema = app.openapi()
    check = schema["paths"]["/check"]["post"]
    verify = schema["paths"]["/v1/verify"]["post"]
    job = schema["paths"]["/v1/jobs/{job_id}"]["get"]
    assert "202" in check["responses"]
    assert "200" in check["responses"]
    assert "200" in verify["responses"]
    assert "200" in job["responses"]
    for operation in (check, verify, job):
        for code in ("400", "401", "402", "413", "429", "500"):
            assert code in operation["responses"]
        names = {item.get("BearerAuth") is not None or item.get("ApiKeyAuth") is not None or item == {} for item in operation["security"]}
        assert True in names or operation["security"]
        assert any("BearerAuth" in item or "ApiKeyAuth" in item or item == {} for item in operation["security"])
    schemes = schema["components"]["securitySchemes"]
    assert schemes["BearerAuth"]["scheme"] == "bearer"
    assert schemes["ApiKeyAuth"]["name"] == "X-API-Key"
    blob = json.dumps(schema)
    assert "CheckReportOut" in blob or "claim" in blob
    assert "VerificationOut" in blob


def test_limit_responses_send_retry_after_and_every_response_has_a_request_id(caplog):
    service, _, _, cfg = make_service(free_daily_spend_cap_usd=0)
    app = create_app(settings=cfg, service=service)
    caplog.set_level(logging.INFO, logger="factcheck")
    with TestClient(app) as client:
        blocked = client.post(
            "/v1/verify",
            json={"text": "The council raised taxes."},
            headers={"X-Client-Fingerprint": "cap", "X-Request-Id": "req-cap-1"},
        )
        assert blocked.status_code == 429
        assert blocked.headers["retry-after"]
        assert blocked.headers["x-request-id"] == "req-cap-1"
        assert blocked.json()["error"]["retry_after"] == int(blocked.headers["retry-after"])
        assert blocked.json()["request_id"] == "req-cap-1"
        health = client.get("/health")
        assert health.headers["x-request-id"]
        assert health.json()["request_id"] == health.headers["x-request-id"]
    assert any("request_id=req-cap-1" in record.message for record in caplog.records)

    service, _, _, cfg = make_service()
    app = create_app(settings=cfg, service=service)
    with TestClient(app) as client:
        headers = {"X-Client-Fingerprint": "once"}
        first = client.post("/check", json={"text": "17 * 24 = 408"}, headers=headers)
        assert first.status_code == 200
        second = client.post("/check", json={"text": "2 + 2 = 4"}, headers=headers)
        assert second.status_code == 402
        assert second.headers["retry-after"]
        assert second.json()["error"]["retry_after"] == int(second.headers["retry-after"])
        assert second.json()["request_id"]


def test_mcp_start_check_returns_a_job_the_get_job_tool_can_poll():
    from factcheck.mcp_server import set_identity
    from tests.conftest import client

    service, _, _, _ = make_service()
    mcp = build_mcp(service)
    set_identity(client(fingerprint="mcp-async"))
    started = asyncio.run(mcp.call_tool("start_check", {"text": "17 * 24 = 408"}))
    accepted = json.loads(started.content[0].text)
    assert accepted["status"] == "queued"
    job_id = accepted["job_id"]
    payload = {}
    for _ in range(40):
        polled = asyncio.run(mcp.call_tool("get_job", {"job_id": job_id}))
        payload = json.loads(polled.content[0].text)
        if payload["status"] in {"done", "error"}:
            break
    assert payload["status"] == "done"
    assert "progress" in payload
    assert payload["report"]["claims"][0]["claim"]
    set_identity(client(fingerprint="mcp-block"))
    blocking = asyncio.run(mcp.call_tool("verify_text", {"text": "2 + 2 = 4"}))
    verified = json.loads(blocking.content[0].text)
    assert verified["claims"][0]["text"]
