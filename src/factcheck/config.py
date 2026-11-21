"""Runtime settings. Environment variables use the field name in upper case."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from factcheck.branding import SERVICE_NAME


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = SERVICE_NAME
    database_url: str = "sqlite:///./data/factcheck.db"
    redis_url: str = ""
    cache_backend: str = "auto"

    public_base_url: str = "http://127.0.0.1:8741"
    upgrade_url: str = ""
    host: str = "0.0.0.0"
    port: int = 8741

    llm_mode: str = "auto"
    llm_provider_order: str = "chutes,openai_compat,soma"
    request_timeout_seconds: float = 45.0

    chutes_api_key: str = ""
    chutes_base_url: str = "https://llm.chutes.ai/v1"
    # Verified against GET https://llm.chutes.ai/v1/models on 2026-11-21.
    # All three are confidential_compute TEE models from different families.
    judge_a_model: str = "deepseek-ai/DeepSeek-V3.2-TEE"
    judge_b_model: str = "Qwen/Qwen3.5-397B-A17B-TEE"
    tiebreak_model: str = "zai-org/GLM-5.1-TEE"
    extract_model: str = "deepseek-ai/DeepSeek-V3.2-TEE"

    # Soma is an internal dev endpoint. It is not a production provider.
    soma_api_key: str = ""
    soma_base_url: str = "https://app.thesoma.ai/v1"
    soma_model: str = "openrouter/deepseek/deepseek-v4-pro"
    soma_enabled: bool = False
    soma_production: bool = False

    openai_compat_base_url: str = ""
    openai_compat_api_key: str = ""
    openai_compat_model: str = ""

    desearch_api_key: str = ""
    desearch_base_url: str = "https://api.desearch.ai"
    google_factcheck_api_key: str = ""
    semantic_scholar_api_key: str = ""
    openalex_email: str = ""
    openalex_api_key: str = ""
    ncbi_api_key: str = ""
    ncbi_email: str = "dev@localhost"

    free_daily_spend_cap_usd: float = Field(
        default=20.0,
        validation_alias=AliasChoices("FREE_DAILY_SPEND_CAP_USD", "FREE_DAILY_CAP_USD"),
    )
    free_check_reserve_usd: float = 0.25
    credits_per_check: float = 1.0
    max_text_chars: int = 20_000
    max_input_claims: int = 20
    max_output_claims: int = 24
    max_checked_claims: int = 6
    check_concurrency: int = 4
    request_budget_seconds: float = 40
    retrieval_timeout_seconds: float = 8
    stale_after_days: int = 120
    cache_ttl_seconds: int = 86_400
    cache_ttl_time_sensitive_seconds: int = 1_800
    cache_ttl_opinion_seconds: int = 21_600
    low_evidence_strength: float = 0.45

    bootstrap_api_key: str = ""
    bootstrap_credits: float = 0.0
    owner_ips: str = ""
    owner_key: str = ""

    # USD per million tokens. Unknown models use the conservative default.
    default_input_usd_per_million: float = 1.0
    default_output_usd_per_million: float = 3.0

    model_prices: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            # Chutes catalog snapshot, 2026-11-21, USD per 1M input/output tokens.
            "deepseek-ai/DeepSeek-V3.2-TEE": (1.0, 1.0),
            "deepseek-ai/DeepSeek-V4-Flash-0731-TEE": (0.44, 1.32),
            "Qwen/Qwen3.5-397B-A17B-TEE": (0.45, 3.0),
            "Qwen/Qwen3.5-27B-TEE": (0.2, 1.75),
            "zai-org/GLM-5.1-TEE": (0.9, 2.9),
            "zai-org/GLM-5.2-TEE": (1.25, 3.95),
            "mock-a/heuristic": (0.1, 0.1),
            "mock-b/heuristic": (0.1, 0.1),
            "mock-tie/heuristic": (0.1, 0.1),
            "openrouter/deepseek/deepseek-v4-pro": (0.85, 2.65),
        }
    )

    def resolved_upgrade_url(self) -> str:
        if self.upgrade_url:
            return self.upgrade_url
        return self.public_base_url.rstrip("/") + "/upgrade"

    def price_for(self, model: str) -> tuple[float, float]:
        return self.model_prices.get(
            model,
            (self.default_input_usd_per_million, self.default_output_usd_per_million),
        )


def get_settings() -> Settings:
    return Settings()
