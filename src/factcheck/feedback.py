"""Caller reports of a bad verdict. Stored as JSONL. The constitution is not changed."""

from __future__ import annotations

import json
import logging
import threading
import time
from hashlib import sha256
from uuid import uuid4

from factcheck.claim_cache import normalize_claim
from factcheck.spend import data_path

logger = logging.getLogger("factcheck")

PUBLIC = {"supported", "hypothesis", "opinion", "conflicting", "unverified"}
_LOCK = threading.Lock()
_MEMORY: list[dict] = []
KEYED_PER_HOUR = 30
ANON_PER_HOUR = 3


class FeedbackError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


def caller_label(identity) -> str:
    if getattr(identity, "agent_name", ""):
        return "agent:" + identity.agent_name
    if getattr(identity, "tester", False):
        return "tester:" + (identity.tester_hash or "")[:8]
    if getattr(identity, "owner", False):
        return "owner"
    if getattr(identity, "api_key", None):
        return "key:" + sha256(identity.api_key.encode()).hexdigest()[:8]
    return "anon:" + sha256((getattr(identity, "ip", "") or "unknown").encode()).hexdigest()[:8]


def _rows(settings) -> list[dict]:
    path = data_path(settings, "feedback.jsonl")
    if path is None or not path.exists():
        return list(_MEMORY)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _within_hour(row: dict) -> bool:
    try:
        return time.time() - float(row.get("created_unix") or 0) < 3600
    except (TypeError, ValueError):
        return False


def record_feedback(settings, identity, *, request_id: str, claim: str, suggested_verdict: str, reason: str, source_url: str = "") -> dict:
    request_id = (request_id or "").strip()
    claim = " ".join((claim or "").split())
    verdict = (suggested_verdict or "").strip().lower()
    reason = (reason or "").strip()
    source_url = (source_url or "").strip()
    if not request_id or not claim or not reason:
        raise FeedbackError(400, "invalid_request", "request_id, claim, and reason are required.")
    if verdict not in PUBLIC:
        raise FeedbackError(400, "invalid_request", "suggested_verdict must be one of the five public verdicts.")
    if source_url and not source_url.startswith(("http://", "https://")):
        raise FeedbackError(400, "invalid_request", "source_url must be an http(s) URL.")
    if len(reason) > 2000 or len(claim) > 2000:
        raise FeedbackError(400, "invalid_request", "The feedback is too long.")
    caller = caller_label(identity)
    keyed = not caller.startswith("anon:")
    limit = KEYED_PER_HOUR if keyed else ANON_PER_HOUR
    normalized = normalize_claim(claim)
    with _LOCK:
        rows = _rows(settings)
        recent = [row for row in rows if row.get("caller") == caller and _within_hour(row)]
        if len(recent) >= limit:
            raise FeedbackError(429, "rate_limited", "Too many feedback reports from this caller.")
        for row in rows:
            if row.get("request_id") == request_id and row.get("claim_key") == normalized and row.get("caller") == caller:
                return {"feedback_id": row["feedback_id"], "duplicate": True}
        item = {
            "feedback_id": str(uuid4()),
            "request_id": request_id,
            "claim": claim,
            "claim_key": normalized,
            "suggested_verdict": verdict,
            "reason": reason,
            "source_url": source_url,
            "caller": caller,
            "created_unix": time.time(),
        }
        _MEMORY.append(item)
        path = data_path(settings, "feedback.jsonl")
        if path is not None:
            with path.open("a") as handle:
                handle.write(json.dumps(item) + "\n")
    logger.info(
        "feedback id=%s request_id=%s caller=%s suggested=%s",
        item["feedback_id"],
        request_id,
        caller,
        verdict,
    )
    return {"feedback_id": item["feedback_id"], "duplicate": False}
