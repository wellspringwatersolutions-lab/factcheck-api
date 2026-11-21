from factcheck.billing import BillingError
from factcheck.storage import new_api_key

from tests.conftest import client, make_service


class _CostRetriever:
    def __init__(self):
        self.cost_usd = 0.0
        self.warnings: list[str] = []

    def search(self, query: str, *, limit: int = 6, scientific: bool = False, categories: list[str] | None = None):
        self.cost_usd += 0.5
        return []


def test_free_check_is_once_per_fingerprint():
    service, _, _, _ = make_service()
    first = service.verify(text=None, claims=["17 * 24 = 408"], client=client(fingerprint="a"))
    assert first.usage.tier == "free"
    try:
        service.verify(text=None, claims=["2 + 2 = 4"], client=client(fingerprint="a"))
        raise AssertionError("second free check should fail")
    except BillingError as exc:
        assert exc.status == 402
        assert exc.code == "free_tier_exhausted"
        assert exc.message.startswith("Free check used. More checks:")
    other = service.verify(text=None, claims=["2 + 2 = 4"], client=client(fingerprint="b"))
    assert other.claims[0].verdict == "Supported"


def test_spend_cap_blocks_the_next_free_check_without_consuming_it():
    service, _, _, _ = make_service(
        retriever=_CostRetriever(),
        free_daily_spend_cap_usd=0.6,
        free_check_reserve_usd=0.5,
    )
    claim = "The council raised property taxes in 2019 after a long debate."
    first = service.verify(text=None, claims=[claim], client=client(fingerprint="cap-a"))
    assert first.usage.estimated_cost_usd >= 0.5
    try:
        service.verify(text=None, claims=[claim], client=client(fingerprint="cap-b"))
        raise AssertionError("cap should block")
    except BillingError as exc:
        assert exc.status == 429
        assert exc.code == "spend_cap_exceeded"
    # The blocked client did not spend their one free check.
    service.settings.free_daily_spend_cap_usd = 20
    allowed = service.verify(text=None, claims=["2 + 2 = 4"], client=client(fingerprint="cap-b"))
    assert allowed.claims[0].verdict == "Supported"


def test_credits_are_prepaid_and_rejected_when_empty():
    service, _, factory, _ = make_service(credits_per_check=1)
    raw = new_api_key(factory, credits=1, label="test")
    paid = client(api_key=raw)
    service.verify(text=None, claims=["2 + 2 = 4"], client=paid)
    try:
        service.verify(text=None, claims=["2 + 2 = 5"], client=paid)
        raise AssertionError("balance should be empty")
    except BillingError as exc:
        assert exc.status == 402
        assert exc.code == "insufficient_credits"
    try:
        service.verify(text=None, claims=["2 + 2 = 4"], client=client(api_key="fc_nope"))
        raise AssertionError("bad key")
    except BillingError as exc:
        assert exc.status == 401


def test_failed_check_does_not_consume_the_free_check():
    service, _, _, _ = make_service()

    class BoomExtract:
        def extract(self, text, claims):
            raise RuntimeError("boom")

    service.extractor = BoomExtract()
    try:
        service.verify(text="The council raised property taxes in 2019 after a long debate.", claims=None, client=client(fingerprint="retry"))
        raise AssertionError("expected failure")
    except RuntimeError:
        pass
    service.extractor = __import__("factcheck.extract", fromlist=["HeuristicExtractor"]).HeuristicExtractor()
    result = service.verify(text=None, claims=["2 + 2 = 4"], client=client(fingerprint="retry"))
    assert result.verification_id
