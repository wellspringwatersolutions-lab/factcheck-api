import logging
import time

from fastapi.testclient import TestClient

from factcheck.app import create_app
from factcheck.cleanup import peel_leading_question
from factcheck.orchestrator import request_budget

from tests.conftest import client, make_service


def test_a_failing_extractor_still_retrieves_and_ignores_the_question_line(caplog):
    service, retriever, _, cfg = make_service()

    class Boom:
        def extract(self, text, claims):
            raise ValueError('bad json {"claims":')

    service.extractor = Boom()
    pasted = "Was the stockpile plan sound?\n\nThe Pentagon gave commanders 90 days to harden bases after the hack."
    question, answer = peel_leading_question(pasted)
    assert question == "Was the stockpile plan sound?"
    assert "Pentagon" in answer
    assert request_budget(cfg, pasted) == 45
    assert request_budget(cfg, "x" * 3900) == 75
    caplog.set_level(logging.WARNING, logger="factcheck")
    started = time.monotonic()
    result = service.verify(text=pasted, claims=None, client=client(), charge=False, started_at=started)
    assert retriever.queries
    assert result.warnings
    assert any(item.startswith("extraction_fallback:ValueError") for item in result.warnings)
    assert all("stockpile plan" not in claim.text.lower() for claim in result.claims)
    assert any("Pentagon" in claim.text or "commanders" in claim.text for claim in result.claims)
    assert any("bad json" in rec.message for rec in caplog.records)
    assert any(rec.message.startswith("check phases extract=") for rec in caplog.records)


def test_a_slow_extractor_is_cut_off_and_retrieval_still_runs():
    service, retriever, _, _ = make_service()

    class Slow:
        def extract(self, text, claims):
            time.sleep(30)
            return [], []

    service.extractor = Slow()
    started = time.monotonic()
    result = service.verify(
        text="The Pentagon gave commanders 90 days to harden bases after the hack.",
        claims=None,
        client=client(),
        charge=False,
        started_at=started,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 30
    assert retriever.queries
    assert result.claims
    assert any(item.startswith("extraction_fallback:TimeoutError") for item in result.warnings)


def test_a_question_stays_inside_the_budget_when_chutes_is_slow():
    from factcheck.review import TwoPassReview

    service, _, _, cfg = make_service()

    def slow_draft(question):
        time.sleep(3)
        return ("mock", "John Adams was the second president. He was a Founding Father.", "no")

    class Slow:
        def extract(self, text, claims):
            time.sleep(30)
            return [], []

        def retry_keypoints(self, text):
            time.sleep(30)
            return []

    class Sleepy:
        def complete(self, **kwargs):
            time.sleep(30)
            raise TimeoutError("review")

    service.drafter = slow_draft
    service.extractor = Slow()
    service.reviewer = TwoPassReview(Sleepy(), cfg)
    started = time.monotonic()
    result = service.verify(
        text="Was John Adams a criminal?",
        claims=None,
        client=client(),
        charge=False,
        budget=8,
        started_at=started,
    )
    assert time.monotonic() - started < 10
    assert result.synopsis.rating == "Not enough evidence"
    assert result.synopsis.answer == "not_enough_evidence"
    assert result.claims
    assert all(claim.effort == "skipped" for claim in result.claims)


def test_async_check_returns_a_job_the_site_can_poll():
    service, _, _, cfg = make_service()
    app = create_app(settings=cfg, service=service)
    with TestClient(app) as http:
        created = http.post(
            "/check",
            json={"text": "The Pentagon gave commanders 90 days to harden bases.", "async": True},
            headers={"X-Client-Fingerprint": "job-a"},
        )
        assert created.status_code == 202, created.text
        body = created.json()
        assert body["status"] == "queued"
        job_id = body["job_id"]
        report = None
        for _ in range(50):
            polled = http.get(f"/v1/jobs/{job_id}")
            assert polled.status_code == 200
            payload = polled.json()
            assert "progress" in payload and "done" in payload["progress"] and "total" in payload["progress"]
            if payload["status"] == "done":
                report = payload["report"]
                break
            time.sleep(0.05)
        assert report is not None
        assert report["claims"]
        prefer = http.post(
            "/check",
            json={"text": "17 * 24 = 408"},
            headers={"Prefer": "respond-async", "X-Client-Fingerprint": "job-b"},
        )
        assert prefer.status_code == 202
        missing = http.get("/v1/jobs/does-not-exist")
        assert missing.status_code == 404
