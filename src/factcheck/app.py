"""HTTP API and the mounted MCP server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from urllib.parse import urlparse

import asyncio
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from mcp.server.transport_security import TransportSecuritySettings

from factcheck.billing import BillingError, ClientIdentity, owner_ips, owner_key_matches, resolve_client_ip
from factcheck.constitution import constitution_document, constitution_markdown
from factcheck.questions import DraftError
from factcheck.config import Settings, get_settings
from factcheck.discovery import llms_txt, manifest
from factcheck.jobs import enqueue_check
from factcheck.mcp_server import build_mcp, current_identity, reset_identity, set_identity
from factcheck.payments import StripeCreditPacks, TaoDeposits
from factcheck.publish import publish_verification
from factcheck.report_html import render_home, render_report
from factcheck.schemas import (
    CheckReportOut,
    ErrorResponse,
    JobAccepted,
    JobStatusOut,
    PublicVerificationOut,
    VerifyIn,
)
from factcheck.web_report import to_check_report
from factcheck.wiring import build_service

logger = logging.getLogger("factcheck")

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "The request body is missing or invalid."},
    401: {"model": ErrorResponse, "description": "The API key is unknown."},
    402: {"model": ErrorResponse, "description": "The free check was used, or the key has no credits. Retry-After is set."},
    413: {"model": ErrorResponse, "description": "The text or the claim list is too long."},
    429: {"model": ErrorResponse, "description": "The daily spend cap was reached. Retry-After is set."},
    500: {"model": ErrorResponse, "description": "The check failed before a verdict was saved."},
    503: {"model": ErrorResponse, "description": "The draft answer could not be written."},
}


class RequestIdMiddleware:
    """Put X-Request-Id on every response and log it. Never log a key."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers") or []}
        request_id = (headers.get("x-request-id") or "").strip() or str(uuid4())
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id
        status_holder = {"status": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                raw = [item for item in list(message.get("headers") or []) if item[0].lower() != b"x-request-id"]
                raw.append((b"x-request-id", request_id.encode()))
                message["headers"] = raw
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                "request_id=%s method=%s path=%s status=%s",
                request_id,
                scope.get("method"),
                scope.get("path"),
                status_holder["status"],
            )


def _request_id(request: Request) -> str:
    state = request.scope.get("state") or {}
    if isinstance(state, dict) and state.get("request_id"):
        return str(state["request_id"])
    return str(getattr(getattr(request, "state", None), "request_id", "") or "")


def _error(request: Request, status: int, code: str, message: str, retry_after: int | None = None, upgrade_url: str | None = None):
    rid = _request_id(request)
    error = {"code": code, "message": message, "request_id": rid}
    if upgrade_url:
        error["upgrade_url"] = upgrade_url
    if retry_after is not None:
        error["retry_after"] = retry_after
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(status_code=status, content={"error": error, "request_id": rid}, headers=headers)


def _stamp(request: Request, payload: dict) -> dict:
    rid = _request_id(request)
    body = dict(payload)
    body["request_id"] = rid
    error = body.get("error")
    if isinstance(error, dict):
        error = dict(error)
        error["request_id"] = rid
        body["error"] = error
    return body


class IdentityMiddleware:
    def __init__(self, app, settings: Settings | None = None):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        client = scope.get("client") or ("unknown", 0)
        auth = headers.get("authorization", "")
        api_key = auth[7:].strip() if auth.lower().startswith("bearer ") else headers.get("x-api-key")
        ip = resolve_client_ip(headers, str(client[0]))
        configured_ips = owner_ips(self.settings.owner_ips) if self.settings else set()
        owner = ip in configured_ips or owner_key_matches(headers.get("x-owner-key"), self.settings.owner_key if self.settings else "")
        identity = ClientIdentity(
            ip=ip,
            user_agent=headers.get("user-agent", ""),
            fingerprint_header=headers.get("x-client-fingerprint", ""),
            api_key=api_key or None,
            owner=owner,
        )
        token = set_identity(identity)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_identity(token)


def _security(settings: Settings) -> TransportSecuritySettings:
    host = urlparse(settings.public_base_url).hostname or "127.0.0.1"
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", "testserver:*", f"{host}:*", host],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://testserver:*",
            f"http://{host}:*",
            f"https://{host}:*",
        ],
    )


def create_app(settings: Settings | None = None, service=None) -> FastAPI:
    settings = settings or get_settings()
    if settings.llm_mode == "auto" and not any(
        [settings.chutes_api_key, settings.soma_api_key, settings.openai_compat_api_key]
    ):
        settings.llm_mode = "mock"
    service = service or build_service(settings)
    logger.warning(
        "google fact-check retriever %s",
        "enabled" if settings.google_factcheck_api_key else "disabled",
    )
    mcp = build_mcp(service)
    mcp_app = mcp.streamable_http_app(stateless_http=True, transport_security=_security(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title=settings.service_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service
    app.add_middleware(IdentityMiddleware, settings=settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://wellspringwatersolutions-lab.github.io"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIdMiddleware)

    @app.get("/health")
    def health(request: Request):
        body = service.health()
        body["request_id"] = _request_id(request)
        return body

    @app.get(
        "/constitution",
        response_model=None,
        responses={200: {"description": "The constitution as markdown, or JSON when Accept is application/json."}},
    )
    def constitution(request: Request, format: str = ""):
        wants_json = format == "json" or "application/json" in (request.headers.get("accept") or "")
        if wants_json:
            return _stamp(request, constitution_document())
        return PlainTextResponse(constitution_markdown(), media_type="text/markdown")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return render_home(settings.public_base_url)

    @app.get("/llms.txt", response_class=PlainTextResponse)
    def llms():
        return llms_txt(settings.public_base_url)

    @app.get("/.well-known/mcp.json")
    def well_known():
        return manifest(settings.public_base_url)

    @app.get("/upgrade", response_class=HTMLResponse)
    def upgrade():
        url = settings.resolved_upgrade_url()
        body = (
            f"<h1>More checks</h1><p>Free check used. More checks: {url}</p>"
            "<p>Credit packs are not on sale yet. Stripe and TAO payments are stubs.</p>"
        )
        return render_home(settings.public_base_url).replace("<h1>", body + "<h1>", 1)

    @app.post("/v1/billing/checkout")
    def checkout(pack_id: str = "pack_100", provider: str = "stripe"):
        gateway = TaoDeposits() if provider == "tao" else StripeCreditPacks()
        intent = gateway.create_checkout(api_key_id="", pack_id=pack_id)
        return JSONResponse(status_code=501, content={"error": {"code": "payment_not_configured", **intent.__dict__}})

    def _limit(body: VerifyIn, request: Request):
        question = (body.question or "").strip() or None
        if not body.text and not body.claims and not question:
            return _error(request, 400, "invalid_request", "Provide text, claims, or a question.")
        if (body.text and len(body.text) > settings.max_text_chars) or (question and len(question) > settings.max_text_chars):
            return _error(request, 413, "payload_too_large", "Text is too long.")
        if body.claims and len(body.claims) > settings.max_input_claims:
            return _error(request, 413, "payload_too_large", "Too many claims.")
        return None

    def _run_verify(body: VerifyIn, request: Request, disconnect_check=None, *, started_at=None, budget=None, on_progress=None):
        rejected = _limit(body, request)
        if rejected is not None:
            return rejected
        question = (body.question or "").strip() or None
        identity = current_identity()
        try:
            return service.verify(
                text=body.text,
                claims=body.claims,
                client=identity,
                disconnect_check=disconnect_check,
                question=question,
                started_at=started_at if started_at is not None else time.monotonic(),
                budget=budget,
                on_progress=on_progress,
            )
        except BillingError as exc:
            return _error(request, exc.status, exc.code, exc.message, retry_after=exc.retry_after, upgrade_url=exc.upgrade_url)
        except DraftError as exc:
            payload = exc.body()
            message = payload.get("error", {}).get("message") if isinstance(payload, dict) else "The draft answer could not be written."
            return _error(request, 503, "draft_failed", message or "The draft answer could not be written.")
        except Exception:
            logger.exception("request_id=%s check failed", _request_id(request))
            return _error(request, 500, "internal_error", "The check failed before a verdict was saved.")

    async def _execute(body: VerifyIn, request: Request):
        disconnected = asyncio.Event()

        async def watch() -> None:
            try:
                while not disconnected.is_set():
                    if await request.is_disconnected():
                        disconnected.set()
                        return
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                return

        watcher = asyncio.create_task(watch())
        try:
            return await asyncio.to_thread(_run_verify, body, request, disconnected.is_set)
        finally:
            watcher.cancel()

    @app.post(
        "/v1/verify",
        responses={200: {"model": PublicVerificationOut, "description": "Verification. claims[].verdict is supported, hypothesis, opinion, conflicting, or unverified, in that order."}, **_ERROR_RESPONSES},
        summary="Check text or claims. Each claim uses the field text.",
    )
    async def verify(body: VerifyIn, request: Request):
        """Blocking check. The claim wording is `claims[].text`, not `claim`."""
        result = await _execute(body, request)
        if isinstance(result, JSONResponse):
            return result
        return _stamp(request, publish_verification(result))

    def _wants_async(body: VerifyIn, request: Request) -> bool:
        prefer = (request.headers.get("prefer") or "").lower()
        return bool(body.async_job or "respond-async" in prefer)

    def _start_job(body: VerifyIn, request: Request):
        rejected = _limit(body, request)
        if rejected is not None:
            return rejected
        accepted = enqueue_check(
            service,
            text=body.text,
            claims=body.claims,
            question=(body.question or "").strip() or None,
            identity=current_identity(),
            budget=120,
        )
        return JSONResponse(status_code=202, content=_stamp(request, accepted))

    @app.post(
        "/check",
        responses={
            200: {"model": CheckReportOut, "description": "Site report. Each claim uses the field claim."},
            202: {"model": JobAccepted, "description": "Async check accepted. Poll GET /v1/jobs/{job_id}."},
            **_ERROR_RESPONSES,
        },
        summary="Check text for the public site. Each claim uses the field claim.",
    )
    async def check(body: VerifyIn, request: Request):
        """Blocking report, or 202 when async is true or Prefer is respond-async."""
        if _wants_async(body, request):
            return _start_job(body, request)
        result = await _execute(body, request)
        if isinstance(result, JSONResponse):
            return result
        source_text = (body.question or "").strip() or body.text or "\n".join(body.claims or [])
        return _stamp(request, to_check_report(result, source_text))

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobStatusOut,
        responses={404: {"model": ErrorResponse, "description": "No job with that id."}, **_ERROR_RESPONSES},
        summary="Poll an async check.",
    )
    def get_job(job_id: str, request: Request):
        """status is queued, running, done, or error. report uses the /check shape when done."""
        found = service.store.get_job(job_id)
        if found is None:
            return _error(request, 404, "not_found", "No job with that id.")
        return _stamp(request, found)

    @app.get("/v1/verification/{verification_id}", response_model=PublicVerificationOut, responses={404: {"model": ErrorResponse}})
    def get_verification(verification_id: str, request: Request):
        found = service.get(verification_id)
        if found is None:
            return _error(request, 404, "not_found", "No verification with that id.")
        return _stamp(request, publish_verification(found))

    @app.get("/v1/verification/{verification_id}/report", response_class=HTMLResponse)
    def report(verification_id: str):
        found = service.get(verification_id)
        if found is None:
            return HTMLResponse("<p>No verification with that id.</p>", status_code=404)
        return render_report(found)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(request: Request, exc: RequestValidationError):
        del exc
        return _error(request, 400, "invalid_request", "The request body is not valid.")

    def openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=settings.service_name,
            version="5",
            description=(
                "Fact-checking for agents. Send a paid key as Authorization: Bearer or X-API-Key. "
                "The first check is free. POST /v1/verify uses claims[].text. POST /check uses claims[].claim."
            ),
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        components["securitySchemes"] = {
            "BearerAuth": {"type": "http", "scheme": "bearer", "description": "Authorization: Bearer <api key>"},
            "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
        }
        for path, methods in schema.get("paths", {}).items():
            if path not in {"/check", "/v1/verify", "/v1/jobs/{job_id}", "/v1/verification/{verification_id}"}:
                continue
            for operation in methods.values():
                if isinstance(operation, dict):
                    operation["security"] = [{}, {"BearerAuth": []}, {"ApiKeyAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = openapi
    app.mount("/", mcp_app)
    return app
