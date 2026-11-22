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

The rules are published at {root}/constitution (markdown, or JSON with `Accept: application/json`).

## Auth

The first check from a client is free and needs no key. Send a paid key as `Authorization: Bearer <key>` or `X-API-Key: <key>`. Do not put the key in the URL.

## Time limits

A normal check aims to finish within 45 seconds. Text longer than 3,500 characters is allowed 75 seconds. An async job is allowed 120 seconds. The clock starts after the body is parsed. Extraction stops at 22 seconds. The query writer waits 8 seconds. A check stops at 60 Desearch calls.

## MCP

{install}

## Tools

- verify_text: blocking check of a passage
- verify_claims: blocking check of a list of claims
- start_check: start a check and return job_id without waiting
- get_job: poll status, progress, and the report for that job_id
- get_verification: fetch a stored result by id

## HTTP

POST {root}/v1/verify
POST {root}/check
GET {root}/v1/jobs/{{job_id}}
GET {root}/v1/verification/{{id}}
GET {root}/v1/verification/{{id}}/report
GET {root}/constitution

Both POST bodies accept `text`, `claims`, and `question`. `POST /check` with `"async": true`, or the header `Prefer: respond-async`, returns `202` `{{"job_id", "status": "queued"}}`. Poll `GET /v1/jobs/{{job_id}}` for `status` (`queued`, `running`, `done`, `error`), `progress` (`done`, `total`), and `report` when done.

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
        "tools": ["verify_text", "verify_claims", "start_check", "get_job", "get_verification"],
        "docs": f"{root}/llms.txt",
        "constitution": f"{root}/constitution",
        "disclaimer": DISCLAIMER,
    }
