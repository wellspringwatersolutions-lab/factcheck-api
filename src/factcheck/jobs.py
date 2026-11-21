"""Background checks shared by POST /check and the MCP start_check tool."""

from __future__ import annotations

import logging
import threading
import time
from uuid import uuid4

from factcheck.billing import BillingError
from factcheck.questions import DraftError
from factcheck.web_report import to_check_report

logger = logging.getLogger("factcheck")


def enqueue_check(service, *, text, claims, question, identity, budget: float = 120) -> dict:
    """Start a check and return immediately. The caller polls GET /v1/jobs/{id}."""
    job_id = str(uuid4())
    service.store.create_job(job_id)
    source_text = (question or "").strip() or text or "\n".join(claims or [])

    def progress(done: int, total: int) -> None:
        service.store.update_job(job_id, status="running", done=done, total=total)

    def run() -> None:
        from factcheck.mcp_server import reset_identity, set_identity

        token = set_identity(identity)
        service.store.update_job(job_id, status="running")
        try:
            result = service.verify(
                text=text,
                claims=claims,
                client=identity,
                question=question,
                started_at=time.monotonic(),
                budget=budget,
                on_progress=progress,
            )
        except BillingError as exc:
            logger.warning("job %s failed %s", job_id, exc.code)
            service.store.update_job(job_id, status="error", error=exc.code)
            return
        except DraftError:
            logger.warning("job %s failed draft_failed", job_id)
            service.store.update_job(job_id, status="error", error="draft_failed")
            return
        except Exception as exc:  # noqa: BLE001 - the job record is the error channel
            logger.warning("job %s failed %s", job_id, exc.__class__.__name__)
            service.store.update_job(job_id, status="error", error=exc.__class__.__name__)
            return
        finally:
            reset_identity(token)
        report = to_check_report(result, source_text)
        total = len(report.get("claims") or [])
        service.store.update_job(job_id, status="done", done=total, total=total, report=report)

    threading.Thread(target=run, name=f"factcheck-job-{job_id[:8]}", daemon=True).start()
    return {"job_id": job_id, "status": "queued"}
