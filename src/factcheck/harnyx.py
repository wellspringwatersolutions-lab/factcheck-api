"""Harnyx (Bittensor subnet 67) as a backup page reader.

The public API is a research task, not a fetch-by-URL call. It needs an
approved account and HARNYX_API_KEY. A completed task currently costs 0.10
credits. This adapter is a no-op when the key is unset. It does not create an
account.
"""

from __future__ import annotations

import hashlib
import logging
import time

import httpx

logger = logging.getLogger("factcheck")

API = "https://api.harnyx.ai/v1/research/tasks"


class HarnyxReader:
    def __init__(self, api_key: str, client: httpx.Client | None = None, wait_seconds: float = 6.0):
        self.api_key = (api_key or "").strip()
        self.client = client or httpx.Client(timeout=8.0)
        self.wait_seconds = wait_seconds

    def read(self, url: str) -> str:
        if not self.api_key or not url:
            return ""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Idempotency-Key": hashlib.sha256(url.encode()).hexdigest(),
            "Content-Type": "application/json",
        }
        try:
            created = self.client.post(API, headers=headers, json={"input": f"Quote the main text of this page: {url}"})
        except httpx.HTTPError as exc:
            logger.warning("harnyx create failed %s", exc.__class__.__name__)
            return ""
        if created.status_code >= 400:
            logger.warning("harnyx create status=%s", created.status_code)
            return ""
        payload = created.json() if created.content else {}
        task_id = payload.get("id") or ""
        deadline = time.monotonic() + self.wait_seconds
        while task_id and time.monotonic() < deadline:
            if payload.get("status") in {"completed", "failed"}:
                break
            time.sleep(0.2)
            try:
                follow = self.client.get(f"{API}/{task_id}", headers={"Authorization": f"Bearer {self.api_key}"})
            except httpx.HTTPError as exc:
                logger.warning("harnyx poll failed %s", exc.__class__.__name__)
                return ""
            if follow.status_code >= 400:
                logger.warning("harnyx poll status=%s", follow.status_code)
                return ""
            payload = follow.json() if follow.content else {}
        return _excerpts(payload, url)


def _excerpts(payload: dict, url: str) -> str:
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        return ""
    pieces = []
    for citation in payload.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        cited = citation.get("url") or ""
        if url.split("#")[0].rstrip("/") not in cited and cited not in url:
            continue
        for excerpt in citation.get("excerpts") or []:
            if excerpt:
                pieces.append(str(excerpt))
    if pieces:
        return "\n".join(pieces)[:4000]
    output = payload.get("output")
    if isinstance(output, str):
        return output[:4000]
    return ""
