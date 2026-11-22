"""Free trial keys. No signup, three checks, one key per IP per UTC day. Only the hash is stored."""

from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
from hashlib import sha256

from factcheck.spend import data_path, sha256_hex

CHECKS = 3


class TrialError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class TrialBook:
    def __init__(self, settings):
        self.path = data_path(settings, "trial_keys.json")
        self.keys: dict[str, int] = {}
        self.issued: dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()

    def digests(self) -> list[str]:
        with self._lock:
            return list(self.keys)

    def remaining(self, digest: str) -> int:
        with self._lock:
            return int(self.keys.get(digest, 0))

    def issue(self, ip: str) -> tuple[str, int]:
        today = datetime.now(timezone.utc).date().isoformat()
        ip_hash = sha256((ip or "unknown").encode()).hexdigest()
        with self._lock:
            if self.issued.get(ip_hash) == today:
                raise TrialError(429, "trial_rate_limited", "This network already received a trial key today.")
            raw = "fc_trial_" + secrets.token_urlsafe(24)
            digest = sha256_hex(raw)
            self.keys[digest] = CHECKS
            self.issued[ip_hash] = today
            self._save()
        return raw, CHECKS

    def consume(self, digest: str) -> int:
        with self._lock:
            left = max(0, int(self.keys.get(digest, 0)) - 1)
            self.keys[digest] = left
            self._save()
            return left

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        for digest, remaining in (payload.get("keys") or {}).items():
            try:
                self.keys[str(digest)] = int(remaining)
            except (TypeError, ValueError):
                continue
        for ip_hash, day in (payload.get("issued") or {}).items():
            self.issued[str(ip_hash)] = str(day)

    def _save(self) -> None:
        if self.path is None:
            return
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"keys": self.keys, "issued": self.issued}))
        temporary.replace(self.path)
