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

{DISCLAIMER}

## MCP

{install}

## Tools

- verify_text: check a passage
- verify_claims: check a list of claims
- get_verification: fetch a stored result by id

## HTTP

POST {root}/v1/verify
GET {root}/v1/verification/{{id}}
GET {root}/v1/verification/{{id}}/report

One free check per client. More checks: {root}/upgrade
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
        "tools": ["verify_text", "verify_claims", "get_verification"],
        "docs": f"{root}/llms.txt",
        "disclaimer": DISCLAIMER,
    }
