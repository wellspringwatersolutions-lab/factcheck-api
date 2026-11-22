"""Nightly known-answer run. It records misses. It does not edit the constitution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from factcheck.feedback import _rows as feedback_rows
from factcheck.publish import map_verdict
from factcheck.spend import data_path

LABELS = ["supported", "hypothesis", "opinion", "conflicting", "unverified"]
CAP_USD = 0.30
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "accuracy_set.json"


def load_set(path: Path | None = None) -> list[dict]:
    rows = json.loads((path or FIXTURE).read_text())
    if len(rows) != 12:
        raise ValueError("the accuracy set must contain 12 claims")
    for row in rows:
        if row.get("expected") not in LABELS or not row.get("text") or not row.get("id"):
            raise ValueError(f"bad accuracy row: {row.get('id')}")
    return rows


def _public(result) -> str:
    if not result.claims:
        return "unverified"
    claim = result.claims[0]
    verdict, _detail, _label = map_verdict(claim.verdict, claim.claim_type, claim.verdict_detail)
    return verdict or "unverified"


def _blank_confusion() -> dict:
    return {expected: {got: 0 for got in LABELS} for expected in LABELS}


def run_accuracy(service, settings, *, cap: float = CAP_USD, path: Path | None = None) -> dict:
    spent = 0.0
    matched = 0
    checked = 0
    failures = []
    confusion = _blank_confusion()
    aborted = False
    for row in load_set(path):
        if spent >= cap:
            aborted = True
            break
        result = service.verify(text=row["text"], claims=None, client=None, charge=False)
        cost = float(getattr(result.usage, "estimated_cost_usd", 0.0) or 0.0)
        spent = round(spent + cost, 6)
        got = _public(result)
        expected = row["expected"]
        confusion[expected][got] = confusion[expected].get(got, 0) + 1
        checked += 1
        if got == expected:
            matched += 1
        else:
            failures.append({"id": row["id"], "expected": expected, "got": got, "text": row["text"]})
        if spent > cap:
            aborted = True
            break
    record = {
        "date": datetime.now(timezone.utc).isoformat(),
        "accuracy": round(matched / checked, 4) if checked else 0.0,
        "checked": checked,
        "matched": matched,
        "confusion": confusion,
        "cost_usd": spent,
        "aborted": aborted,
        "failures": failures,
    }
    _append_run(settings, record)
    build_proposals(settings, record)
    return record


def _runs_path(settings):
    return data_path(settings, "accuracy_runs.json")


def _append_run(settings, record: dict) -> None:
    path = _runs_path(settings)
    runs = load_runs(settings)
    runs.append(record)
    if path is None:
        _MEMORY_RUNS[:] = runs[-14:]
        return
    path.write_text(json.dumps(runs[-30:]))


_MEMORY_RUNS: list[dict] = []


def load_runs(settings) -> list[dict]:
    path = _runs_path(settings)
    if path is None or not path.exists():
        return list(_MEMORY_RUNS)
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return rows if isinstance(rows, list) else []


def public_report(settings) -> dict:
    runs = load_runs(settings)[-14:]
    latest = runs[-1] if runs else None
    return {
        "date": None if latest is None else latest["date"],
        "accuracy": None if latest is None else latest["accuracy"],
        "confusion": None if latest is None else latest["confusion"],
        "cost_usd": None if latest is None else latest["cost_usd"],
        "runs": [
            {"date": row["date"], "accuracy": row["accuracy"], "cost_usd": row["cost_usd"], "checked": row["checked"]}
            for row in runs
        ],
    }


def build_proposals(settings, record: dict | None = None) -> list[dict]:
    """Suggestions only. Nothing here is written into the constitution."""
    if record is None:
        runs = load_runs(settings)
        record = runs[-1] if runs else {"failures": []}
    proposals = []
    for failure in record.get("failures") or []:
        proposals.append(
            {
                "source": "accuracy",
                "id": failure["id"],
                "suggestion": (
                    f"The known answer for '{failure['text']}' is {failure['expected']}, "
                    f"and the run returned {failure['got']}. Review the rule for that label. Do not apply this automatically."
                ),
            }
        )
    for row in feedback_rows(settings):
        proposals.append(
            {
                "source": "feedback",
                "id": row.get("feedback_id"),
                "suggestion": (
                    f"A caller suggested {row.get('suggested_verdict')} for '{row.get('claim')}'. "
                    f"Reason: {row.get('reason')}. Do not apply this automatically."
                ),
            }
        )
    path = data_path(settings, "proposals.json")
    if path is not None:
        path.write_text(json.dumps(proposals))
    return proposals
