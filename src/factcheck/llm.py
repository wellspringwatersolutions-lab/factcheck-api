"""OpenAI-compatible providers with failover. Soma is not a production provider."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from factcheck.config import Settings
from factcheck.textutil import parse_json_object


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    family: str


def family_of(model: str) -> str:
    text = (model or "").lower()
    for name in ("deepseek", "qwen", "glm", "mock-a", "mock-b", "mock-tie"):
        if name in text:
            return name
    return text.split("/")[0] or "unknown"


class OpenAICompatClient:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        production: bool = True,
        fixed_model: str | None = None,
        timeout: float = 45.0,
        client: httpx.Client | None = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.production = production
        self.fixed_model = fixed_model
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 800) -> LLMResult:
        use_model = self.fixed_model or model
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            # Qwen and GLM otherwise spend the budget on hidden reasoning and can return empty content.
            "enable_thinking": False,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        response = self._post(url, payload, headers)
        if response.status_code == 400:
            for key in ("chat_template_kwargs", "enable_thinking", "reasoning_effort", "response_format"):
                payload.pop(key, None)
            response = self._post(url, payload, headers)
        response.raise_for_status()
        content, usage, served = self._read(response)
        if not content.strip():
            retry = self._post(url, payload, headers)
            retry.raise_for_status()
            content, usage, served = self._read(retry)
        if not content.strip():
            content = '{"passages": []}'
        return LLMResult(
            content=content,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            model=served or use_model,
            provider=self.name,
            family=family_of(served or use_model),
        )

    def _post(self, url: str, payload: dict, headers: dict) -> httpx.Response:
        return self._client.post(url, json=payload, headers=headers)

    @staticmethod
    def _read(response: httpx.Response) -> tuple[str, dict, str]:
        data = response.json()
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        content = message.get("content") or ""
        return content, data.get("usage") or {}, data.get("model") or ""


class FailoverClient:
    def __init__(self, providers: list):
        self.providers = providers

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> LLMResult:
        errors: list[str] = []
        for provider in self.providers:
            try:
                return provider.complete(model=model, messages=messages, temperature=temperature)
            except Exception as exc:  # noqa: BLE001 - failover is the point
                errors.append(f"{provider.name}: {exc}")
        raise LLMError("; ".join(errors) or "no language-model provider is configured")


class ScriptedLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> LLMResult:
        self.calls.append({"model": model, "messages": messages})
        content = self.responses.pop(0) if self.responses else "{}"
        return LLMResult(content, 20, 20, model, "script", family_of(model))


def build_llm(settings: Settings):
    if settings.llm_mode == "mock":
        return None
    providers = []
    order = [part.strip() for part in settings.llm_provider_order.split(",") if part.strip()]
    available = {
        "chutes": bool(settings.chutes_api_key),
        "openai_compat": bool(settings.openai_compat_api_key and settings.openai_compat_base_url),
        "soma": bool(settings.soma_api_key),
    }
    soma_only = available["soma"] and not available["chutes"] and not available["openai_compat"]
    for name in order:
        if name == "chutes" and available["chutes"]:
            providers.append(
                OpenAICompatClient(
                    name="chutes",
                    base_url=settings.chutes_base_url,
                    api_key=settings.chutes_api_key,
                    timeout=settings.request_timeout_seconds,
                )
            )
        elif name == "openai_compat" and available["openai_compat"]:
            providers.append(
                OpenAICompatClient(
                    name="openai_compat",
                    base_url=settings.openai_compat_base_url,
                    api_key=settings.openai_compat_api_key,
                    fixed_model=settings.openai_compat_model or None,
                    timeout=settings.request_timeout_seconds,
                )
            )
        elif name == "soma" and available["soma"] and (settings.soma_enabled or soma_only):
            providers.append(
                OpenAICompatClient(
                    name="soma",
                    base_url=settings.soma_base_url,
                    api_key=settings.soma_api_key,
                    production=False,
                    fixed_model=settings.soma_model,
                    timeout=settings.request_timeout_seconds,
                )
            )
    if settings.llm_mode == "live" and not providers:
        raise LLMError("llm_mode=live but no provider key is configured")
    if not providers:
        return None
    return FailoverClient(providers)


def fetch_chutes_catalog(timeout: float = 10.0) -> list[dict]:
    response = httpx.get("https://llm.chutes.ai/v1/models", timeout=timeout)
    response.raise_for_status()
    return list(response.json().get("data") or [])


def catalog_warnings(settings: Settings, models: list[dict]) -> list[str]:
    by_id = {item.get("id"): item for item in models}
    warnings = []
    for role, model in (
        ("judge A", settings.judge_a_model),
        ("judge B", settings.judge_b_model),
        ("tie-break", settings.tiebreak_model),
        ("extract", settings.extract_model),
    ):
        row = by_id.get(model)
        if row is None:
            warnings.append(f"{role} model {model} is not in the live Chutes catalog")
        elif not row.get("confidential_compute"):
            warnings.append(f"{role} model {model} is not marked confidential_compute")
    families = {family_of(settings.judge_a_model), family_of(settings.judge_b_model), family_of(settings.tiebreak_model)}
    if len(families) < 3:
        warnings.append("judge A, judge B, and the tie-break model should be three different families")
    return warnings


def estimate_cost(settings: Settings, model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = settings.price_for(model)
    return input_tokens / 1_000_000 * pin + output_tokens / 1_000_000 * pout


def loads_object(result: LLMResult) -> dict:
    return parse_json_object(result.content)
