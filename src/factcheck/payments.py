"""Payment stubs. Stripe credit packs and TAO deposits are not wired.

TODO: paid credit packs and checkout for outside agents. Do not build that flow
until the owner asks for it. These classes stay unwired on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass


CREDIT_PACKS = {
    "pack_100": {"credits": 100, "usd": 10},
    "pack_1000": {"credits": 1000, "usd": 80},
}


@dataclass
class CheckoutIntent:
    provider: str
    status: str
    pack_id: str
    credits: int
    message: str


class StripeCreditPacks:
    """Stub. Set STRIPE_SECRET_KEY later; this does not contact Stripe."""

    production_ready = False
    name = "stripe"

    def create_checkout(self, *, api_key_id: str, pack_id: str) -> CheckoutIntent:
        pack = CREDIT_PACKS.get(pack_id)
        credits = pack["credits"] if pack else 0
        return CheckoutIntent(
            provider=self.name,
            status="not_configured",
            pack_id=pack_id,
            credits=credits,
            message="Stripe credit packs are not wired yet.",
        )


class TaoDeposits:
    """Stub for a later TAO deposit flow. This does not move funds."""

    production_ready = False
    name = "tao"

    def create_checkout(self, *, api_key_id: str, pack_id: str) -> CheckoutIntent:
        pack = CREDIT_PACKS.get(pack_id)
        credits = pack["credits"] if pack else 0
        return CheckoutIntent(
            provider=self.name,
            status="not_configured",
            pack_id=pack_id,
            credits=credits,
            message="TAO deposits are not wired yet.",
        )
