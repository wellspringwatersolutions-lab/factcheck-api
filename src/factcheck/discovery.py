"""Discovery documents for agents."""

from __future__ import annotations

from factcheck.branding import SERVICE_NAME
from factcheck.schemas import DISCLAIMER


def llms_txt(base_url: str) -> str:
    root = base_url.rstrip("/")
    install = f"add {root}/mcp to Claude/Cursor"
    return f"""# {SERVICE_NAME}

{SERVICE_NAME} checks text for AI agents and researchers.

It extracts atomic claims, recomputes math, retrieves evidence for factual claims, and reports what mainstream fact-checkers say about opinions. A supported verdict is shown only when a retrieved passage or a tool result backs the claim, and the report quotes that passage. Language models rank evidence. They do not decide the verdict.

Public claim verdicts are lowercase and only these five, in this order: supported, hypothesis, opinion, conflicting, unverified. `verdict_label` is the display name, such as Hypothesis. `verdict_detail` is `likely_true` (one good source; still counts toward the rating), `speculative`, or `contradicted` (a conflicting claim the sources go against). Older names such as Supported, likely_true, and Contradicted are accepted on input and mapped to these five on output.

{DISCLAIMER}

The rules are published at {root}/constitution (markdown, or JSON with `Accept: application/json`). `GET /accuracy` publishes the latest known-answer run: date, accuracy, per-label confusion, cost, and the last 14 runs. Failures and feedback become suggestions at `GET /v1/proposals` for the owner. Nothing in that file edits the constitution.

## Auth

The first check from a client is free and needs no key. Send a paid key, a tester key, or the owner key as `Authorization: Bearer <key>` or `X-API-Key: <key>`. The same headers work on the MCP endpoint. Do not put the key in the URL.

A tester key is the sha256 listed in `TESTER_KEY_SHA256S` (unlimited) or `TESTER_KEY_BUDGETS` (`sha256hex:usd`, a lifetime budget; this wins when a hash is in both). Spend is Chutes plus Desearch cost, saved under `DATA_DIR` when that directory is writable, and `TESTER_KEY_SPENT` (`hashprefix8:usd`) is a floor after a redeploy. A check is rejected with HTTP 402 and `tester budget exhausted` when the remaining budget is below the reserve or already used up. The response includes `usage.caller` = `tester` and `usage.tester_budget_remaining_usd`. `GET /v1/tester/budget` returns the same budget. Tester spend still counts toward the daily spend cap. Logs record `caller=tester` and the first 8 characters of the hash, never the key.

`AGENT_KEY_DAILY` is `name:sha256hex:usd_per_day`. It is empty unless an operator sets it; no bot keys are built in. An agent key skips the free-check limit, resets at UTC midnight, and is rejected with HTTP 402 `agent daily budget exhausted` plus `reset_at` when the day's budget is used up. Responses set `usage.caller` to the agent name and `usage.budget_remaining_usd`. Agent spend is saved next to tester spend and still counts toward the daily spend cap.

## Time limits

A normal check aims to finish within 45 seconds. Text longer than 3,500 characters is allowed 75 seconds. An async job is allowed 120 seconds. The clock starts after the body is parsed. Extraction stops at 22 seconds. The query writer waits 8 seconds. A check stops at 60 Desearch calls.

## MCP

{install}

Claude Desktop: add a streamable HTTP server at `{root}/mcp`. Cursor: add `{root}/mcp` to Cursor. Generic streamable HTTP: `POST {root}/mcp`.

`POST {root}/v1/keys/trial` issues one free trial key with no signup: 3 checks for that key, one key per IP per UTC day. The key is stored only as a hash. It works as `Authorization: Bearer` or `X-API-Key` on REST and MCP.

## Tools

- verify_text: blocking check of a passage
- verify_claims: blocking check of a list of claims
- start_check: start a check and return job_id without waiting
- get_job: poll status, progress, and the report for that job_id
- get_verification: fetch a stored result by id
- report_bad_verdict: report a wrong verdict. Does not edit the constitution.

## HTTP

POST {root}/v1/verify
POST {root}/check
GET {root}/v1/jobs/{{job_id}}
GET {root}/v1/jobs/{{job_id}}/events
GET {root}/v1/verification/{{id}}
GET {root}/v1/verification/{{id}}/report
POST {root}/v1/feedback
GET {root}/constitution
GET {root}/accuracy

Both POST bodies accept `text`, `claims`, and `question`. `POST /check` with `"async": true`, or the header `Prefer: respond-async`, returns `202` `{{"job_id", "status": "queued"}}`. Poll `GET /v1/jobs/{{job_id}}` for `status` (`queued`, `running`, `done`, `error`), `progress` (`done`, `total`), `claims` as each one finishes, and `report` when done. `GET /v1/jobs/{{job_id}}/events` streams those claims as server-sent events (`event: claim`, then `event: done`). `usage.cached_claims` counts verdicts reused from the claim cache.

`POST /v1/verify` and `POST /check` return different claim shapes. Do not mix them up.

- `POST /v1/verify` returns a verification object. Each claim's wording is the field `text`. The object also has `verification_id`, `claims`, `mode`, `draft_answer`, `synopsis`, `review`, and `usage`.
- `POST /check` returns the site report. Each claim's wording is the field `claim` (not `text`). The object has `input_text`, `claims`, `mode`, `draft_answer`, `synopsis`, `review`, and `usage`. It does not include `verification_id`. The async job's `report` uses this same `/check` shape.

Errors use `{{"error": {{"code", "message", "request_id"}}, "request_id"}}`. Codes: 400 invalid body, 401 unknown key, 402 free check or credits exhausted, 413 text or claim list too long, 429 spend cap, 500 check failed, 503 draft failed. 402 and 429 also send `Retry-After` and `error.retry_after` in seconds. Every response sends `X-Request-Id`, and JSON bodies include `request_id`.

One free check per client. More checks: {root}/upgrade
Paid credit packs and checkout for outside agents are not available yet.
"""


def manifest(base_url: str) -> dict:
    root = base_url.rstrip("/")
    return {
        "name": SERVICE_NAME,
        "description": (
            "Fact-checking for agents. Supported verdicts require a quoted passage or a tool result. "
            "Opinions are reported as what mainstream fact-checkers say, not as true or false."
        ),
        "mcp_endpoint": f"{root}/mcp",
        "install": f"add {root}/mcp to Claude/Cursor",
        "connect": {
            "claude_desktop": f"Add a streamable HTTP server at {root}/mcp in Claude Desktop.",
            "cursor": f"add {root}/mcp to Cursor",
            "streamable_http": f"POST {root}/mcp",
        },
        "tools": [
            {"name": "verify_text", "description": "Check a passage and return claims with verdict, confidence, and recommended_action."},
            {"name": "verify_claims", "description": "Check a list of claims."},
            {"name": "start_check", "description": "Start a check and return a job id."},
            {"name": "get_job", "description": "Poll a job for progress, claims, and the report."},
            {"name": "get_verification", "description": "Fetch a stored verification by id."},
            {"name": "report_bad_verdict", "description": "Report a wrong verdict. Does not edit the constitution."},
        ],
        "docs": f"{root}/llms.txt",
        "constitution": f"{root}/constitution",
        "disclaimer": DISCLAIMER,
    }
