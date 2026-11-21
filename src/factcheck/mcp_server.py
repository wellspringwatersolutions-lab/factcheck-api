"""Remote MCP server (streamable HTTP) with the same free-tier rules as the REST API."""

from __future__ import annotations

import json
from typing import Annotated

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from factcheck.billing import BillingError, ClientIdentity
from factcheck.branding import SERVICE_NAME
from factcheck.schemas import DISCLAIMER

_IDENTITY = __import__("contextvars").ContextVar("factcheck_identity", default=None)


def current_identity() -> ClientIdentity:
    found = _IDENTITY.get()
    if found is not None:
        return found
    return ClientIdentity(ip="unknown")


def set_identity(identity: ClientIdentity | None):
    return _IDENTITY.set(identity)


def reset_identity(token) -> None:
    _IDENTITY.reset(token)


def _tighten(schema: dict) -> None:
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
    for value in schema.get("properties", {}).values():
        _tighten(value)
    if isinstance(schema.get("items"), dict):
        _tighten(schema["items"])
    for definition in schema.get("$defs", {}).values():
        _tighten(definition)


def build_mcp(service) -> MCPServer:
    instructions = (
        f"{SERVICE_NAME} checks claims against sources and tools. "
        "A supported verdict requires a quoted passage or a tool result such as a recomputed equation. "
        "Opinions return what mainstream fact-checkers say, which is not a true or false label. "
        "Hypotheses are not called true or false. "
        f"{DISCLAIMER} "
        "The first check from a client is free. Send a paid key as Authorization: Bearer <key> or X-API-Key."
    )
    mcp = MCPServer(SERVICE_NAME, instructions=instructions)

    @mcp.tool(
        description=(
            "Extract atomic claims from a passage. Facts are checked against retrieved passages and tools. "
            "Math is recomputed. Opinions return mainstream_check (what fact-checkers say), not true or false. "
            "Supported is returned only when a passage or tool result backs the claim and that quote is included. "
            "With no passage the verdict is Unverified. One free check per client; a paid API key spends credits."
        )
    )
    async def verify_text(
        text: Annotated[str, Field(description="Passage to check", min_length=1, max_length=20000)],
    ) -> str:
        return await _verify(service, text=text, claims=None)

    @mcp.tool(
        description=(
            "Check a list of claims. Each claim is classified (T1 math through T8 opinion). "
            "T1 is recomputed. T8 returns mainstream_check. Other facts require quoted evidence. "
            "Verdicts are Supported, Contradicted, Unverified, Conflicting, Unsupported, or Hypothesis. "
            "Same free-tier rules as verify_text."
        )
    )
    async def verify_claims(
        claims: Annotated[
            list[Annotated[str, Field(min_length=1)]],
            Field(description="Claims to check", min_length=1, max_length=20),
        ],
    ) -> str:
        return await _verify(service, text=None, claims=claims)

    @mcp.tool(description="Fetch a stored verification by id. Does not spend a free check or credits.")
    async def get_verification(
        verification_id: Annotated[str, Field(description="verification_id returned by a check", min_length=8)],
    ) -> str:
        found = await anyio.to_thread.run_sync(service.get, verification_id)
        if found is None:
            raise ToolError(json.dumps({"error": {"code": "not_found", "message": "No verification with that id."}}))
        return json.dumps(found.model_dump(mode="json"), ensure_ascii=False)

    for tool in mcp._tool_manager.list_tools():
        _tighten(tool.parameters)
    return mcp


async def _verify(service, *, text, claims) -> str:
    identity = current_identity()

    def run():
        return service.verify(text=text, claims=claims, client=identity)

    try:
        result = await anyio.to_thread.run_sync(run)
    except BillingError as exc:
        raise ToolError(json.dumps(exc.body())) from exc
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
