"""Local commands: serve, create-key, models."""

from __future__ import annotations

import argparse
import os

import uvicorn

from factcheck.branding import SERVICE_NAME
from factcheck.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=SERVICE_NAME)
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the API and MCP server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    key = sub.add_parser("create-key", help="Create a local API key with prepaid credits")
    key.add_argument("--credits", type=float, default=10)
    key.add_argument("--label", default="local")

    sub.add_parser("models", help="Compare configured Chutes model ids with the live catalog")

    args = parser.parse_args(argv)
    settings = get_settings()
    if args.cmd == "serve":
        uvicorn.run(
            "factcheck.app:create_app",
            factory=True,
            host=args.host or settings.host,
            port=args.port or int(os.environ.get("PORT") or settings.port),
        )
        return 0
    if args.cmd == "create-key":
        from factcheck.storage import init_db, make_engine, make_session_factory, new_api_key

        engine = make_engine(settings.database_url)
        init_db(engine)
        raw = new_api_key(make_session_factory(engine), credits=args.credits, label=args.label)
        print(raw)
        return 0
    if args.cmd == "models":
        from factcheck.llm import catalog_warnings, fetch_chutes_catalog

        models = fetch_chutes_catalog(timeout=settings.request_timeout_seconds)
        print(f"{len(models)} models in the Chutes catalog")
        for warning in catalog_warnings(settings, models):
            print("warning:", warning)
        if not catalog_warnings(settings, models):
            print("configured judge and extract models are present and marked confidential_compute")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
