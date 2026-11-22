"""Verdict cache keyed on the claim text. Memory LRU, with an optional JSON file."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

from factcheck.schemas import ClaimOut

_PUNCT = re.compile(r"[^a-z0-9\s]")
WEEK = 7 * 24 * 3600


def normalize_claim(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


class ClaimCache:
    """Seven-day LRU. A path keeps the same entries across processes."""

    kind = "claim"

    def __init__(self, *, ttl: int = WEEK, limit: int = 512, path: str = ""):
        self.ttl = ttl
        self.limit = limit
        self.path = path
        self._lock = threading.Lock()
        self._items: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        if path:
            self._load()

    def get(self, text: str) -> ClaimOut | None:
        key = normalize_claim(text)
        if not key:
            return None
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires, payload = item
            if expires < time.time():
                self._items.pop(key, None)
                self._save_locked()
                return None
            self._items.move_to_end(key)
            stored = dict(payload)
        return ClaimOut.model_validate(stored)

    def put(self, claim: ClaimOut, *, ttl: int | None = None) -> None:
        if claim.effort == "skipped" or not claim.text:
            return
        key = normalize_claim(claim.text)
        if not key:
            return
        lifetime = self.ttl if ttl is None else ttl
        with self._lock:
            self._items[key] = (time.time() + lifetime, claim.model_dump())
            self._items.move_to_end(key)
            while len(self._items) > self.limit:
                self._items.popitem(last=False)
            self._save_locked()

    def _load(self) -> None:
        file = Path(self.path)
        if not file.exists():
            return
        try:
            rows = json.loads(file.read_text())
        except (OSError, ValueError):
            return
        now = time.time()
        for row in rows if isinstance(rows, list) else []:
            key = row.get("key")
            expires = float(row.get("expires") or 0)
            payload = row.get("payload")
            if key and expires > now and isinstance(payload, dict):
                self._items[key] = (expires, payload)

    def _save_locked(self) -> None:
        if not self.path:
            return
        file = Path(self.path)
        file.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"key": key, "expires": expires, "payload": payload}
            for key, (expires, payload) in self._items.items()
        ]
        temporary = file.with_suffix(file.suffix + ".tmp")
        temporary.write_text(json.dumps(rows))
        temporary.replace(file)
