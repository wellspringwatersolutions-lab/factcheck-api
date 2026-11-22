from fastapi.testclient import TestClient

from factcheck.app import create_app
from factcheck.constitution import constitution_markdown

from tests.conftest import make_service


def test_feedback_is_validated_deduped_and_rate_limited(tmp_path):
    before = constitution_markdown()
    service, _, _, settings = make_service(data_dir=str(tmp_path), owner_key="owner-sample")
    app = create_app(settings=settings, service=service)
    body = {
        "request_id": "req-1",
        "claim": "The Eiffel Tower is in Berlin.",
        "suggested_verdict": "conflicting",
        "reason": "The tower is in Paris.",
        "source_url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
    }
    with TestClient(app) as client:
        bad = client.post("/v1/feedback", json={**body, "suggested_verdict": "Supported"})
        saved = client.post("/v1/feedback", json=body, headers={"X-API-Key": "owner-sample"})
        again = client.post("/v1/feedback", json=body, headers={"Authorization": "Bearer owner-sample"})
        anon = [
            client.post("/v1/feedback", json={**body, "request_id": f"anon-{i}", "claim": f"Claim {i} is wrong."})
            for i in range(4)
        ]
    assert bad.status_code == 400
    assert saved.status_code == 200, saved.text
    assert again.json()["feedback_id"] == saved.json()["feedback_id"]
    assert again.json()["duplicate"] is True
    assert [item.status_code for item in anon] == [200, 200, 200, 429]
    assert constitution_markdown() == before
    assert "Eiffel" in (tmp_path / "feedback.jsonl").read_text()
    assert "owner-sample" not in (tmp_path / "feedback.jsonl").read_text()
