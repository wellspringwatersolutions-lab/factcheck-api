"""HTTP API and the mounted MCP server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from urllib.parse import urlparse

import asyncio
import logging
import threading
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from mcp.server.transport_security import TransportSecuritySettings

from factcheck.billing import BillingError, ClientIdentity, owner_ips, owner_key_matches, resolve_client_ip
from factcheck.questions import DraftError
from factcheck.config import Settings, get_settings
from factcheck.discovery import llms_txt, manifest
from factcheck.mcp_server import build_mcp, current_identity, reset_identity, set_identity
from factcheck.payments import StripeCreditPacks, TaoDeposits
from factcheck.report_html import render_home, render_report
from factcheck.schemas import VerifyIn
from factcheck.web_report import to_check_report
from factcheck.wiring import build_service

logger = logging.getLogger("factcheck")


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

    @app.get("/health")
    def health():
        return service.health()

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

    def _run_verify(body: VerifyIn, disconnect_check=None, *, started_at=None, budget=None, on_progress=None):
        question = (body.question or "").strip() or None
        if not body.text and not body.claims and not question:
            raise HTTPException(status_code=422, detail="Provide text, claims, or a question.")
        if body.text and len(body.text) > settings.max_text_chars:
            raise HTTPException(status_code=422, detail="Text is too long.")
        if question and len(question) > settings.max_text_chars:
            raise HTTPException(status_code=422, detail="Question is too long.")
        if body.claims and len(body.claims) > settings.max_input_claims:
            raise HTTPException(status_code=422, detail="Too many claims.")
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
            return JSONResponse(status_code=exc.status, content=exc.body())
        except DraftError as exc:
            return JSONResponse(status_code=503, content=exc.body())
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "internal_error", "message": "The check failed before a verdict was saved."}},
            )

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
            return await asyncio.to_thread(_run_verify, body, disconnected.is_set)
        finally:
            watcher.cancel()

    @app.post("/v1/verify")
    async def verify(body: VerifyIn, request: Request):
        return await _execute(body, request)

    def _wants_async(body: VerifyIn, request: Request) -> bool:
        prefer = (request.headers.get("prefer") or "").lower()
        return bool(body.async_job or "respond-async" in prefer)

    def _start_job(body: VerifyIn, request: Request):
        job_id = str(uuid4())
        service.store.create_job(job_id)
        identity = current_identity()
        source_text = (body.question or "").strip() or body.text or "\n".join(body.claims or [])

        def progress(done: int, total: int) -> None:
            service.store.update_job(job_id, status="running", done=done, total=total)

        def run() -> None:
            token = set_identity(identity)
            service.store.update_job(job_id, status="running")
            try:
                result = _run_verify(body, None, started_at=time.monotonic(), budget=120, on_progress=progress)
            except Exception as exc:  # noqa: BLE001 - the job record is the error channel
                logger.warning("job %s failed %s", job_id, exc.__class__.__name__)
                service.store.update_job(job_id, status="error", error=exc.__class__.__name__)
                return
            finally:
                reset_identity(token)
            if isinstance(result, JSONResponse):
                service.store.update_job(job_id, status="error", error="check_failed")
                return
            report = to_check_report(result, source_text)
            total = len(report.get("claims") or [])
            service.store.update_job(job_id, status="done", done=total, total=total, report=report)

        threading.Thread(target=run, name=f"factcheck-job-{job_id[:8]}", daemon=True).start()
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})

    @app.post("/check")
    async def check(body: VerifyIn, request: Request):
        if _wants_async(body, request):
            return _start_job(body, request)
        result = await _execute(body, request)
        if isinstance(result, JSONResponse):
            return result
        source_text = (body.question or "").strip() or body.text or "\n".join(body.claims or [])
        return to_check_report(result, source_text)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        found = service.store.get_job(job_id)
        if found is None:
            return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "No job with that id."}})
        return found

    @app.get("/v1/verification/{verification_id}")
    def get_verification(verification_id: str):
        found = service.get(verification_id)
        if found is None:
            return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "No verification with that id."}})
        return found

    @app.get("/v1/verification/{verification_id}/report", response_class=HTMLResponse)
    def report(verification_id: str):
        found = service.get(verification_id)
        if found is None:
            return HTMLResponse("<p>No verification with that id.</p>", status_code=404)
        return render_report(found)

    app.mount("/", mcp_app)
    return app
