from types import SimpleNamespace

from fastapi.testclient import TestClient

from factcheck.accuracy import load_set, run_accuracy
from factcheck.app import create_app
from factcheck.constitution import constitution_markdown

from tests.conftest import make_service


def test_the_accuracy_set_has_twelve_known_answers():
    rows = load_set()
    labels = {row["expected"] for row in rows}
    assert len(rows) == 12
    assert {"supported", "conflicting", "opinion", "hypothesis"} <= labels


def test_a_run_stops_when_the_cost_passes_thirty_cents(tmp_path):
    class Metered:
        def verify(self, **_kwargs):
            return SimpleNamespace(claims=[], usage=SimpleNamespace(estimated_cost_usd=0.2))

    service, _, _, settings = make_service(data_dir=str(tmp_path))
    record = run_accuracy(Metered(), settings, cap=0.30)
    assert record["aborted"] is True
    assert record["checked"] == 2
    assert record["cost_usd"] == 0.4


def test_accuracy_is_public_and_proposals_are_owner_only(tmp_path):
    before = constitution_markdown()
    service, _, _, settings = make_service(data_dir=str(tmp_path), owner_key="owner-sample")
    app = create_app(settings=settings, service=service)
    with TestClient(app) as client:
        empty = client.get("/accuracy")
        denied = client.post("/v1/admin/accuracy-run")
        ran = client.post("/v1/admin/accuracy-run", headers={"Authorization": "Bearer owner-sample"})
        public = client.get("/accuracy")
        secret = client.get("/v1/proposals")
        owned = client.get("/v1/proposals", headers={"X-Owner-Key": "owner-sample"})
    assert empty.status_code == 200
    assert empty.json()["runs"] == []
    assert denied.status_code == 401
    assert ran.status_code == 200, ran.text
    assert ran.json()["checked"] == 12
    assert "supported" in ran.json()["confusion"]
    assert public.json()["accuracy"] == ran.json()["accuracy"]
    assert len(public.json()["runs"]) == 1
    assert secret.status_code == 401
    assert owned.status_code == 200
    assert owned.json()["proposals"]
    assert constitution_markdown() == before
    assert "constitution" not in (tmp_path / "proposals.json").read_text().lower() or "do not apply" in (tmp_path / "proposals.json").read_text().lower()
