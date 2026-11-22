"""SQLAlchemy storage. SQLite is the local default; Postgres is used when DATABASE_URL says so."""

from __future__ import annotations

import hashlib
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import JSON

from factcheck.publish import publish_verification
from factcheck.schemas import VerificationOut


class Base(DeclarativeBase):
    pass


class VerificationRow(Base):
    __tablename__ = "verifications"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict] = mapped_column(JSON)
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tier: Mapped[str] = mapped_column(String(16), default="free")
    api_key_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0)


class ApiKeyRow(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prefix: Mapped[str] = mapped_column(String(16))
    label: Mapped[str] = mapped_column(String(120), default="")
    credits: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CreditLedgerRow(Base):
    __tablename__ = "credit_ledger"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    api_key_id: Mapped[str] = mapped_column(String(36), index=True)
    delta: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class JobRow(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String(400), nullable=True)


class FreeCheckRow(Base):
    __tablename__ = "free_checks"
    fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    day: Mapped[str] = mapped_column(String(10), index=True)
    status: Mapped[str] = mapped_column(String(16))
    cost_usd: Mapped[float] = mapped_column(Float, default=0)
    verification_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


def make_engine(url: str):
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        else:
            raw = url.removeprefix("sqlite:///")
            if raw and raw != ":memory:":
                Path(raw).parent.mkdir(parents=True, exist_ok=True)
    else:
        kwargs["pool_pre_ping"] = True
    return create_engine(url, **kwargs)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def new_api_key(factory, *, credits: float, label: str = "") -> str:
    from uuid import uuid4

    raw = "fc_" + secrets.token_urlsafe(24)
    with factory() as session:
        session.add(
            ApiKeyRow(
                id=str(uuid4()),
                key_hash=hash_key(raw),
                prefix=raw[:10],
                label=label,
                credits=credits,
                created_at=datetime.now(timezone.utc),
            )
        )
        session.commit()
    return raw


class Store:
    def __init__(self, factory):
        self.factory = factory
        self._job_lock = threading.Lock()

    def save(self, payload: VerificationOut, *, fingerprint: str | None, tier: str, api_key_id: str | None) -> None:
        with self.factory() as session:
            session.add(
                VerificationRow(
                    id=payload.verification_id,
                    created_at=datetime.fromisoformat(payload.created_at),
                    payload=publish_verification(payload),
                    fingerprint=fingerprint,
                    tier=tier,
                    api_key_id=api_key_id,
                    estimated_cost_usd=payload.usage.estimated_cost_usd,
                )
            )
            session.commit()

    def create_job(self, job_id: str) -> None:
        now = datetime.now(timezone.utc)
        with self.factory() as session:
            session.add(JobRow(id=job_id, status="queued", created_at=now, updated_at=now, progress_done=0, progress_total=0))
            session.commit()

    def update_job(self, job_id: str, *, status: str | None = None, done: int | None = None, total: int | None = None, report: dict | None = None, error: str | None = None) -> None:
        with self._job_lock, self.factory() as session:
            row = session.get(JobRow, job_id)
            if row is None:
                return
            if status is not None:
                row.status = status
            if done is not None:
                row.progress_done = done
            if total is not None:
                row.progress_total = total
            if report is not None:
                row.report = report
            if error is not None:
                row.error = error[:400]
            row.updated_at = datetime.now(timezone.utc)
            session.commit()

    def set_job_claims(self, job_id: str, claims: list[dict], *, done: int, total: int) -> None:
        """Store claims that have finished. A finished job keeps its full report."""
        with self._job_lock, self.factory() as session:
            row = session.get(JobRow, job_id)
            if row is None or row.status == "done":
                return
            row.status = "running"
            row.progress_done = done
            row.progress_total = total
            row.report = {"claims": claims}
            row.updated_at = datetime.now(timezone.utc)
            session.commit()

    def get_job(self, job_id: str) -> dict | None:
        with self.factory() as session:
            row = session.get(JobRow, job_id)
            if row is None:
                return None
            raw = row.report if isinstance(row.report, dict) else None
            claims = list((raw or {}).get("claims") or [])
            report = raw if row.status == "done" else None
            return {
                "job_id": row.id,
                "status": row.status,
                "progress": {"done": row.progress_done, "total": row.progress_total},
                "claims": claims,
                "report": report,
                "error": row.error,
            }

    def get(self, verification_id: str) -> VerificationOut | None:
        with self.factory() as session:
            row = session.get(VerificationRow, verification_id)
            if row is None:
                return None
            return VerificationOut.model_validate(row.payload)


class MemoryCache:
    kind = "memory"

    def __init__(self):
        self._data: dict[str, tuple[float, dict]] = {}

    def get(self, key: str):
        import time

        item = self._data.get(key)
        if not item:
            return None
        expires, value = item
        if expires < time.time():
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: dict, ttl: int) -> None:
        import time

        self._data[key] = (time.time() + ttl, value)


class RedisCache:
    kind = "redis"

    def __init__(self, url: str):
        import redis

        self.redis = redis.Redis.from_url(url, socket_timeout=2)

    def get(self, key: str):
        import json

        raw = self.redis.get("factcheck:" + key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: dict, ttl: int) -> None:
        import json

        self.redis.setex("factcheck:" + key, ttl, json.dumps(value))

    def ping(self) -> bool:
        return bool(self.redis.ping())


def build_cache(settings):
    if settings.cache_backend == "memory" or not settings.redis_url:
        return MemoryCache()
    if settings.cache_backend == "redis" or settings.cache_backend == "auto":
        try:
            cache = RedisCache(settings.redis_url)
            cache.ping()
            return cache
        except Exception:
            if settings.cache_backend == "redis":
                raise
            return MemoryCache()
    return MemoryCache()
