"""HTTP API and the mounted MCP server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from mcp.server.transport_security import TransportSecuritySettings

from factcheck.billing import BillingError, ClientIdentity
from factcheck.config import Settings, get_settings
from factcheck.discovery import llms_txt, manifest
from factcheck.mcp_server import build_mcp, current_identity, reset_identity, set_identity
from factcheck.payments import StripeCreditPacks, TaoDeposits
from factcheck.report_html import render_home, render_report
from factcheck.schemas import VerifyIn
from factcheck.web_report import to_check_report
from factcheck.wiring import build_service


class IdentityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        client = scope.get("client") or ("unknown", 0)
        auth = headers.get("authorization", "")
        api_key = auth[7:].strip() if auth.lower().startswith("bearer ") else headers.get("x-api-key")
        identity = ClientIdentity(
            ip=str(client[0]),
            user_agent=headers.get("user-agent", ""),
            fingerprint_header=headers.get("x-client-fingerprint", ""),
            api_key=api_key or None,
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
    mcp = build_mcp(service)
    mcp_app = mcp.streamable_http_app(stateless_http=True, transport_security=_security(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title=settings.service_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service
    app.add_middleware(IdentityMiddleware)
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

    def _run_verify(body: VerifyIn):
        if not body.text and not body.claims:
            raise HTTPException(status_code=422, detail="Provide text or claims.")
        if body.text and len(body.text) > settings.max_text_chars:
            raise HTTPException(status_code=422, detail="Text is too long.")
        if body.claims and len(body.claims) > settings.max_input_claims:
            raise HTTPException(status_code=422, detail="Too many claims.")
        identity = current_identity()
        try:
            return service.verify(text=body.text, claims=body.claims, client=identity)
        except BillingError as exc:
            return JSONResponse(status_code=exc.status, content=exc.body())
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "internal_error", "message": "The check failed before a verdict was saved."}},
            )

    @app.post("/v1/verify")
    def verify(body: VerifyIn):
        return _run_verify(body)

    @app.post("/check")
    def check(body: VerifyIn):
        result = _run_verify(body)
        if isinstance(result, JSONResponse):
            return result
        source_text = body.text or "\n".join(body.claims or [])
        return to_check_report(result, source_text)

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
