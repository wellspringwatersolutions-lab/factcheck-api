"""One free check per client, a daily spend cap, and prepaid credits."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select

from factcheck.storage import ApiKeyRow, CreditLedgerRow, FreeCheckRow, hash_key


class BillingError(Exception):
    def __init__(self, code: str, message: str, status: int, upgrade_url: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.upgrade_url = upgrade_url

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "upgrade_url": self.upgrade_url}}


@dataclass
class ClientIdentity:
    ip: str
    user_agent: str = ""
    fingerprint_header: str = ""
    api_key: str | None = None

    @property
    def fingerprint(self) -> str:
        material = "\n".join([self.ip.strip(), self.user_agent.strip(), self.fingerprint_header.strip()])
        return hashlib.sha256(material.encode()).hexdigest()


@dataclass
class Reservation:
    kind: str
    fingerprint: str | None = None
    api_key_id: str | None = None


class BillingGate:
    def __init__(self, factory, settings):
        self.factory = factory
        self.settings = settings
        self._lock = threading.Lock()

    def _upgrade(self, kind: str) -> str:
        url = self.settings.resolved_upgrade_url()
        if kind == "free":
            return f"Free check used. More checks: {url}"
        if kind == "cap":
            return f"Free check daily spend cap reached. More checks: {url}"
        if kind == "credits":
            return f"Insufficient credits. More checks: {url}"
        return f"More checks: {url}"

    def authorize(self, client: ClientIdentity) -> Reservation:
        if client.api_key:
            return self._authorize_paid(client.api_key)
        return self._authorize_free(client)

    def _authorize_paid(self, raw_key: str) -> Reservation:
        with self._lock, self.factory() as session:
            row = session.scalar(select(ApiKeyRow).where(ApiKeyRow.key_hash == hash_key(raw_key)))
            if row is None:
                raise BillingError("invalid_api_key", "Unknown API key.", 401, self.settings.resolved_upgrade_url())
            if row.credits + 1e-9 < self.settings.credits_per_check:
                raise BillingError("insufficient_credits", self._upgrade("credits"), 402, self.settings.resolved_upgrade_url())
            return Reservation(kind="paid", api_key_id=row.id)

    def _authorize_free(self, client: ClientIdentity) -> Reservation:
        today = datetime.now(timezone.utc).date().isoformat()
        fingerprint = client.fingerprint
        reserve = self.settings.free_check_reserve_usd
        cap = self.settings.free_daily_spend_cap_usd
        with self._lock, self.factory() as session:
            spent = session.scalar(
                select(func.coalesce(func.sum(FreeCheckRow.cost_usd), 0.0)).where(
                    FreeCheckRow.day == today, FreeCheckRow.status == "completed"
                )
            ) or 0.0
            inflight = session.scalar(
                select(func.count()).select_from(FreeCheckRow).where(FreeCheckRow.day == today, FreeCheckRow.status == "reserved")
            ) or 0
            if float(spent) + inflight * reserve + reserve > cap + 1e-9:
                raise BillingError("spend_cap_exceeded", self._upgrade("cap"), 429, self.settings.resolved_upgrade_url())
            existing = session.get(FreeCheckRow, fingerprint)
            if existing is not None:
                created = existing.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - created).total_seconds()
                stale = existing.status == "reserved" and age > 180
                if stale:
                    session.delete(existing)
                    session.commit()
                else:
                    raise BillingError("free_tier_exhausted", self._upgrade("free"), 402, self.settings.resolved_upgrade_url())
            session.add(
                FreeCheckRow(
                    fingerprint=fingerprint,
                    created_at=datetime.now(timezone.utc),
                    day=today,
                    status="reserved",
                    cost_usd=0,
                )
            )
            session.commit()
        return Reservation(kind="free", fingerprint=fingerprint)

    def complete(self, reservation: Reservation | None, cost_usd: float, verification_id: str) -> None:
        if reservation is None:
            return
        if reservation.kind == "free" and reservation.fingerprint:
            with self._lock, self.factory() as session:
                row = session.get(FreeCheckRow, reservation.fingerprint)
                if row is not None:
                    row.status = "completed"
                    row.cost_usd = float(cost_usd)
                    row.verification_id = verification_id
                    row.day = datetime.now(timezone.utc).date().isoformat()
                    session.commit()
            return
        if reservation.kind == "paid" and reservation.api_key_id:
            with self._lock, self.factory() as session:
                row = session.get(ApiKeyRow, reservation.api_key_id)
                if row is None or row.credits + 1e-9 < self.settings.credits_per_check:
                    raise BillingError("insufficient_credits", self._upgrade("credits"), 402, self.settings.resolved_upgrade_url())
                row.credits -= self.settings.credits_per_check
                session.add(
                    CreditLedgerRow(
                        api_key_id=row.id,
                        delta=-self.settings.credits_per_check,
                        reason=f"verification:{verification_id}",
                        created_at=datetime.now(timezone.utc),
                    )
                )
                session.commit()

    def release(self, reservation: Reservation | None) -> None:
        if reservation is None or reservation.kind != "free" or not reservation.fingerprint:
            return
        with self._lock, self.factory() as session:
            row = session.get(FreeCheckRow, reservation.fingerprint)
            if row is not None and row.status == "reserved":
                session.delete(row)
                session.commit()


class NoBilling:
    def authorize(self, client):
        return Reservation(kind="eval")

    def complete(self, reservation, cost_usd, verification_id):
        return None

    def release(self, reservation):
        return None
