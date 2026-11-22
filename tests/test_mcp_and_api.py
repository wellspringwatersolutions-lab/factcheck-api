import asyncio
import json

from fastapi.testclient import TestClient

from factcheck.app import create_app
from factcheck.mcp_server import build_mcp

from tests.conftest import make_service


def test_mcp_tool_schemas():
    service, _, _, _ = make_service()
    mcp = build_mcp(service)
    tools = asyncio.run(mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {"verify_text", "verify_claims", "start_check", "get_job", "get_verification", "report_bad_verdict"}
    for tool in tools:
        schema = tool.model_dump(mode="json")["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"]
        assert tool.description
    text_schema = by_name["verify_text"].model_dump(mode="json")["input_schema"]
    assert "text" in text_schema["properties"]
    claims_schema = by_name["verify_claims"].model_dump(mode="json")["input_schema"]
    assert claims_schema["properties"]["claims"]["type"] == "array"
    assert "passage" in by_name["verify_text"].description.lower() or "Unverified" in by_name["verify_text"].description


def test_http_free_tier_report_and_discovery():
    service, _, _, settings = make_service()
    app = create_app(settings=settings, service=service)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["service"]
        home = client.get("/")
        assert home.status_code == 200
        assert "add http://127.0.0.1:8741/mcp to Claude/Cursor" in home.text
        manifest = client.get("/.well-known/mcp.json").json()
        assert manifest["install"].endswith("/mcp to Claude/Cursor")
        llms = client.get("/llms.txt").text
        assert "cannot prove that no better evidence exists" in llms

        headers = {"X-Client-Fingerprint": "http-a"}
        response = client.post(
            "/v1/verify",
            json={"claims": ["17 * 24 = 408", "I think the bill is a disaster for small farms."]},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["claims"][0]["verdict"] == "supported"
        assert body["claims"][0]["verdict_label"] == "Supported"
        assert body["claims"][0]["verdict_detail"] is None
        assert body["claims"][1]["verdict"] == "opinion"
        assert body["claims"][1]["mainstream_check"]["label"] == "What mainstream fact-checkers say"
        assert "upgrade_url" in body
        report = client.get(f"/v1/verification/{body['verification_id']}/report")
        assert report.status_code == 200
        assert "claim-fact" in report.text
        assert "claim-opinion" in report.text
        assert "What mainstream fact-checkers say" in report.text
        assert "strength of support" in report.text
        assert "cannot prove that no better evidence exists" in report.text
        again = client.post("/v1/verify", json={"claims": ["2 + 2 = 4"]}, headers=headers)
        assert again.status_code == 402
        assert again.json()["error"]["code"] == "free_tier_exhausted"


def test_check_matches_website_schema_and_sends_cors():
    service, _, _, settings = make_service()
    app = create_app(settings=settings, service=service)
    origin = "https://wellspringwatersolutions-lab.github.io"
    with TestClient(app) as client:
        preflight = client.options(
            "/check",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )
        assert preflight.headers.get("access-control-allow-origin") == origin
        response = client.post(
            "/check",
            json={"text": "17 * 24 = 408"},
            headers={"Origin": origin, "X-Client-Fingerprint": "site"},
        )
        assert response.status_code == 200, response.text
        assert response.headers.get("access-control-allow-origin") == origin
        body = response.json()
        assert body["input_text"] == "17 * 24 = 408"
        claim = body["claims"][0]
        assert claim["verdict"] == "supported"
        assert claim["claim_type"] == "fact"
        assert claim["claim_category"]
        source = claim["sources"][0]
        assert {"url", "title", "domain", "checked_on", "quote", "source_type"} <= set(source)
        assert source["source_type"] == "tool_result"


def test_scifact_sample_is_stratified():
    from factcheck.eval.runner import load_examples

    rows = load_examples("scifact", 15)
    labels = [row["label"] for row in rows]
    assert len(set(labels)) > 1


def test_get_verification_tool_returns_json():
    service, _, _, _ = make_service()
    saved = service.verify(text=None, claims=["2 + 2 = 4"], client=__import__("tests.conftest", fromlist=["client"]).client(), charge=False)
    mcp = build_mcp(service)
    result = asyncio.run(mcp.call_tool("get_verification", {"verification_id": saved.verification_id}))
    payload = json.loads(result.content[0].text)
    assert payload["verification_id"] == saved.verification_id
