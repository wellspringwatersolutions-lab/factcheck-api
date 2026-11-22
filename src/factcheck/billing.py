"""One free check per client, a daily spend cap, and prepaid credits."""

from __future__ import annotations

import hashlib
import hmac
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select

from factcheck.spend import DailySpendBook, SpendBook, data_path, match_digest, parse_agent_daily, parse_hash_amounts, parse_sha256_list
from factcheck.trial import TrialBook
from factcheck.storage import ApiKeyRow, CreditLedgerRow, FreeCheckRow, hash_key


class BillingError(Exception):
    def __init__(self, code: str, message: str, status: int, upgrade_url: str, retry_after: int | None = None, remaining_usd: float | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.upgrade_url = upgrade_url
        self.retry_after = retry_after
        self.remaining_usd = remaining_usd
        self.reset_at = None

    def body(self) -> dict:
        error = {"code": self.code, "message": self.message, "upgrade_url": self.upgrade_url}
        if self.retry_after is not None:
            error["retry_after"] = self.retry_after
        if self.remaining_usd is not None:
            error["remaining_usd"] = self.remaining_usd
        if self.reset_at:
            error["reset_at"] = self.reset_at
        return {"error": error}


def resolve_client_ip(headers: dict[str, str], peer: str | None) -> str:
    """First X-Forwarded-For hop when a proxy set it, otherwise the socket peer."""
    forwarded = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For") or ""
    hop = forwarded.split(",")[0].strip()
    if hop:
        return hop
    return (peer or "unknown").strip() or "unknown"


def owner_ips(value: str) -> set[str]:
    return {part.strip() for part in (value or "").split(",") if part.strip()}


def owner_key_matches(presented: str | None, expected: str) -> bool:
    """Constant-time check. An empty configured key never matches."""
    if not expected or presented is None or presented == "":
        return False
    left = hashlib.sha256(presented.encode()).digest()
    right = hashlib.sha256(expected.encode()).digest()
    return hmac.compare_digest(left, right)


@dataclass
class ClientIdentity:
    ip: str
    user_agent: str = ""
    fingerprint_header: str = ""
    api_key: str | None = None
    owner: bool = False
    tester: bool = False
    tester_hash: str = ""
    agent_name: str = ""
    agent_hash: str = ""
    trial: bool = False
    trial_hash: str = ""

    @property
    def fingerprint(self) -> str:
        material = "\n".join([self.ip.strip(), self.user_agent.strip(), self.fingerprint_header.strip()])
        return hashlib.sha256(material.encode()).hexdigest()


@dataclass
class Reservation:
    kind: str
    fingerprint: str | None = None
    api_key_id: str | None = None
    tester_hash: str = ""
    agent_hash: str = ""
    trial_hash: str = ""


class BillingGate:
    def __init__(self, factory, settings):
        self.factory = factory
        self.settings = settings
        self._lock = threading.Lock()
        budgets = parse_hash_amounts(getattr(settings, "tester_key_budgets", ""))
        self.tester_budgets = {key: value for key, value in budgets.items() if len(key) == 64}
        unlimited = parse_sha256_list(getattr(settings, "tester_key_sha256s", ""))
        self.tester_unlimited = [item for item in unlimited if item not in self.tester_budgets]
        floors = {key: value for key, value in parse_hash_amounts(getattr(settings, "tester_key_spent", "")).items() if len(key) == 8}
        self.tester_spend = SpendBook(data_path(settings, "tester_spend.json"), floors)
        self.agents = {digest: (name, amount) for name, digest, amount in parse_agent_daily(getattr(settings, "agent_key_daily", ""))}
        self.agent_spend = DailySpendBook(data_path(settings, "agent_spend.json"))
        self.trials = TrialBook(settings)

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
        if getattr(client, "trial", False):
            return self._authorize_trial(client)
        if getattr(client, "agent_name", ""):
            return self._authorize_agent(client)
        if getattr(client, "tester", False):
            return self._authorize_tester(client)
        if getattr(client, "owner", False):
            return self._authorize_owner()
        if client.api_key:
            return self._authorize_paid(client.api_key)
        return self._authorize_free(client)

    def tester_digests(self) -> list[str]:
        return list(self.tester_budgets) + list(self.tester_unlimited)

    def tester_status(self, digest: str) -> dict:
        prefix = digest[:8]
        if digest in self.tester_unlimited:
            return {"caller": "tester", "hash": prefix, "unlimited": True, "budget_usd": None, "spent_usd": 0.0, "remaining_usd": None}
        budget = float(self.tester_budgets.get(digest, 0.0))
        spent = round(self.tester_spend.amount(digest), 6)
        return {
            "caller": "tester",
            "hash": prefix,
            "unlimited": False,
            "budget_usd": budget,
            "spent_usd": spent,
            "remaining_usd": round(budget - spent, 6),
        }

    def _authorize_tester(self, client: ClientIdentity) -> Reservation:
        digest = client.tester_hash
        if digest in self.tester_budgets:
            remaining = self.tester_status(digest)["remaining_usd"]
            minimum = float(self.settings.free_check_reserve_usd)
            if remaining is None or remaining <= 0 or remaining < minimum:
                raise BillingError(
                    "tester_budget_exhausted",
                    "tester budget exhausted",
                    402,
                    self.settings.resolved_upgrade_url(),
                    remaining_usd=remaining,
                )
        reservation = self._authorize_owner(kind="tester")
        reservation.tester_hash = digest
        return reservation

    def agent_status(self, digest: str) -> dict:
        from datetime import datetime, timedelta, timezone

        name, budget = self.agents.get(digest, ("", 0.0))
        spent = round(self.agent_spend.amount(digest), 6)
        tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
        reset = datetime.combine(tomorrow, datetime.min.time(), timezone.utc).isoformat()
        return {
            "caller": name,
            "hash": digest[:8],
            "budget_usd": budget,
            "spent_usd": spent,
            "remaining_usd": round(budget - spent, 6),
            "reset_at": reset,
        }

    def _authorize_agent(self, client: ClientIdentity) -> Reservation:
        digest = client.agent_hash
        status = self.agent_status(digest)
        remaining = status["remaining_usd"]
        minimum = float(self.settings.free_check_reserve_usd)
        if remaining <= 0 or remaining < minimum:
            error = BillingError(
                "agent_daily_budget_exhausted",
                "agent daily budget exhausted",
                402,
                self.settings.resolved_upgrade_url(),
                remaining_usd=remaining,
            )
            error.reset_at = status["reset_at"]
            raise error
        reservation = self._authorize_owner(kind="agent")
        reservation.agent_hash = digest
        return reservation

    def _authorize_trial(self, client: ClientIdentity) -> Reservation:
        left = self.trials.remaining(client.trial_hash)
        if left <= 0:
            raise BillingError("trial_exhausted", "Trial checks are used up.", 402, self.settings.resolved_upgrade_url(), retry_after=3600)
        reservation = self._authorize_owner(kind="trial")
        reservation.trial_hash = client.trial_hash
        return reservation

    def _authorize_paid(self, raw_key: str) -> Reservation:
        with self._lock, self.factory() as session:
            row = session.scalar(select(ApiKeyRow).where(ApiKeyRow.key_hash == hash_key(raw_key)))
            if row is None:
                raise BillingError("invalid_api_key", "Unknown API key.", 401, self.settings.resolved_upgrade_url())
            if row.credits + 1e-9 < self.settings.credits_per_check:
                raise BillingError(
                    "insufficient_credits",
                    self._upgrade("credits"),
                    402,
                    self.settings.resolved_upgrade_url(),
                    retry_after=3600,
                )
            return Reservation(kind="paid", api_key_id=row.id)

    def _spend_would_exceed(self, session, today: str) -> bool:
        reserve = self.settings.free_check_reserve_usd
        cap = self.settings.free_daily_spend_cap_usd
        spent = session.scalar(
            select(func.coalesce(func.sum(FreeCheckRow.cost_usd), 0.0)).where(
                FreeCheckRow.day == today, FreeCheckRow.status == "completed"
            )
        ) or 0.0
        inflight = session.scalar(
            select(func.count()).select_from(FreeCheckRow).where(FreeCheckRow.day == today, FreeCheckRow.status == "reserved")
        ) or 0
        return float(spent) + inflight * reserve + reserve > cap + 1e-9

    def _authorize_owner(self, kind: str = "owner") -> Reservation:
        from uuid import uuid4

        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock, self.factory() as session:
            if self._spend_would_exceed(session, today):
                raise BillingError("spend_cap_exceeded", self._upgrade("cap"), 429, self.settings.resolved_upgrade_url(), retry_after=3600)
            fingerprint = f"{kind}:{uuid4()}"
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
        return Reservation(kind=kind, fingerprint=fingerprint)

    def _authorize_free(self, client: ClientIdentity) -> Reservation:
        today = datetime.now(timezone.utc).date().isoformat()
        fingerprint = client.fingerprint
        with self._lock, self.factory() as session:
            if self._spend_would_exceed(session, today):
                raise BillingError("spend_cap_exceeded", self._upgrade("cap"), 429, self.settings.resolved_upgrade_url(), retry_after=3600)
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
                    raise BillingError(
                        "free_tier_exhausted",
                        self._upgrade("free"),
                        402,
                        self.settings.resolved_upgrade_url(),
                        retry_after=3600,
                    )
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
        if reservation.kind in {"free", "owner", "tester", "agent", "trial"} and reservation.fingerprint:
            with self._lock, self.factory() as session:
                row = session.get(FreeCheckRow, reservation.fingerprint)
                if row is not None:
                    row.status = "completed"
                    row.cost_usd = float(cost_usd)
                    row.verification_id = verification_id
                    row.day = datetime.now(timezone.utc).date().isoformat()
                    session.commit()
            if reservation.kind == "tester" and reservation.tester_hash:
                self.tester_spend.add(reservation.tester_hash, float(cost_usd))
            if reservation.kind == "agent" and reservation.agent_hash:
                self.agent_spend.add(reservation.agent_hash, float(cost_usd))
            if reservation.kind == "trial" and reservation.trial_hash:
                self.trials.consume(reservation.trial_hash)
            return
        if reservation.kind == "paid" and reservation.api_key_id:
            with self._lock, self.factory() as session:
                row = session.get(ApiKeyRow, reservation.api_key_id)
                if row is None or row.credits + 1e-9 < self.settings.credits_per_check:
                    raise BillingError(
                        "insufficient_credits",
                        self._upgrade("credits"),
                        402,
                        self.settings.resolved_upgrade_url(),
                        retry_after=3600,
                    )
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
        if reservation is None or reservation.kind not in {"free", "owner", "tester", "agent", "trial"} or not reservation.fingerprint:
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
