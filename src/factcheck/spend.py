"""Lifetime tester budgets and the on-disk spend ledger.

Keys are matched by the sha256 of the presented secret. The secret is never stored.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from pathlib import Path


def sha256_hex(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _parts(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def parse_sha256_list(value: str) -> list[str]:
    found = []
    for part in _parts(value):
        digest = part.lower()
        if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
            found.append(digest)
    return found


def parse_agent_daily(value: str) -> list[tuple[str, str, float]]:
    """`name:sha256hex:usd_per_day`. No entries are configured by default."""
    found = []
    for part in _parts(value):
        name, separator, rest = part.partition(":")
        digest, separator, raw = rest.partition(":")
        name = name.strip()
        digest = digest.strip().lower()
        if not separator or not name or len(digest) != 64:
            continue
        if not all(char in "0123456789abcdef" for char in digest):
            continue
        try:
            found.append((name, digest, float(raw.strip())))
        except ValueError:
            continue
    return found


def parse_hash_amounts(value: str) -> dict[str, float]:
    """`sha256hex:usd` or `hashprefix8:usd` pairs."""
    found: dict[str, float] = {}
    for part in _parts(value):
        if ":" not in part:
            continue
        left, raw = part.split(":", 1)
        left = left.strip().lower()
        if not left or not all(char in "0123456789abcdef" for char in left):
            continue
        try:
            found[left] = float(raw.strip())
        except ValueError:
            continue
    return found


def match_digest(presented: str | None, digests: list[str]) -> str:
    """Constant-time scan. Returns the matching digest, or ''."""
    if not presented or not digests:
        return ""
    actual = sha256_hex(presented)
    found = ""
    for digest in digests:
        if len(digest) != 64:
            continue
        if hmac.compare_digest(actual, digest):
            found = digest
    return found


def data_path(settings, name: str) -> Path | None:
    """A file under DATA_DIR when that directory exists and is writable."""
    directory = (getattr(settings, "data_dir", "") or os.environ.get("DATA_DIR") or "").strip()
    if not directory:
        return None
    folder = Path(directory)
    if not folder.is_dir() or not os.access(folder, os.W_OK):
        return None
    return folder / name


class SpendBook:
    """Amounts keyed by the full sha256. A prefix seed is a floor, not a replacement."""

    def __init__(self, path: Path | None, floors: dict[str, float] | None = None):
        self.path = path
        self.floors = dict(floors or {})
        self.spent: dict[str, float] = {}
        self._lock = threading.Lock()
        self._load()

    def amount(self, digest: str) -> float:
        stored = float(self.spent.get(digest, 0.0))
        floor = 0.0
        for prefix, value in self.floors.items():
            if digest.startswith(prefix):
                floor = max(floor, value)
        return max(stored, floor)

    def add(self, digest: str, cost: float) -> float:
        with self._lock:
            self.spent[digest] = round(self.amount(digest) + max(0.0, float(cost)), 6)
            self._save()
            return self.spent[digest]

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        spent = payload.get("spent") if isinstance(payload, dict) else None
        if isinstance(spent, dict):
            for key, value in spent.items():
                try:
                    self.spent[str(key).lower()] = float(value)
                except (TypeError, ValueError):
                    continue

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"spent": self.spent}))
        temporary.replace(self.path)


class DailySpendBook:
    """UTC-day totals. A new day starts at zero."""

    def __init__(self, path: Path | None):
        self.path = path
        self.day = ""
        self.spent: dict[str, float] = {}
        self._lock = threading.Lock()
        self._load()

    def amount(self, digest: str) -> float:
        with self._lock:
            self._roll()
            return float(self.spent.get(digest, 0.0))

    def add(self, digest: str, cost: float) -> float:
        with self._lock:
            self._roll()
            self.spent[digest] = round(float(self.spent.get(digest, 0.0)) + max(0.0, float(cost)), 6)
            self._save()
            return self.spent[digest]

    def _today(self) -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).date().isoformat()

    def _roll(self) -> None:
        today = self._today()
        if self.day != today:
            self.day = today
            self.spent = {}

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        self.day = str(payload.get("day") or "")
        spent = payload.get("spent") if isinstance(payload.get("spent"), dict) else {}
        for key, value in spent.items():
            try:
                self.spent[str(key).lower()] = float(value)
            except (TypeError, ValueError):
                continue

    def _save(self) -> None:
        if self.path is None:
            return
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"day": self.day, "spent": self.spent}))
        temporary.replace(self.path)
