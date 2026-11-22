import logging

import httpx
import pytest

from factcheck.llm import LLMError, OpenAICompatClient


def _client(handler, key="sk-live-secret"):
    return OpenAICompatClient(
        name="chutes",
        base_url="https://llm.chutes.ai/v1",
        api_key=key,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _complete(client):
    return client.complete(model="Qwen/Qwen3.5-397B-A17B-TEE", messages=[{"role": "user", "content": "hi"}])


def test_a_timeout_names_the_model_and_hides_the_key(caplog):
    def handler(request):
        raise httpx.TimeoutException("timed out")

    caplog.set_level(logging.WARNING, logger="factcheck.llm")
    with pytest.raises(httpx.TimeoutException):
        _complete(_client(handler))
    assert "chutes failure" in caplog.text
    assert "kind=timeout" in caplog.text
    assert "Qwen/Qwen3.5-397B-A17B-TEE" in caplog.text
    assert "sk-live-secret" not in caplog.text


def test_a_server_error_logs_the_status_and_a_body_snippet(caplog):
    def handler(request):
        return httpx.Response(500, text='{"error":"upstream blew up"}')

    caplog.set_level(logging.WARNING, logger="factcheck.llm")
    with pytest.raises(httpx.HTTPStatusError):
        _complete(_client(handler))
    assert "chutes failure" in caplog.text
    assert "status=500" in caplog.text
    assert "upstream blew up" in caplog.text
    assert "sk-live-secret" not in caplog.text


def test_payment_required_is_logged_as_a_balance_problem(caplog):
    def handler(request):
        return httpx.Response(402, text='{"error":"insufficient_quota"}')

    caplog.set_level(logging.WARNING, logger="factcheck.llm")
    with pytest.raises(httpx.HTTPStatusError):
        _complete(_client(handler))
    assert "chutes balance/quota" in caplog.text
    assert "status=402" in caplog.text
    assert "sk-live-secret" not in caplog.text


def test_a_rate_limit_stays_a_failure_until_the_body_says_quota(caplog):
    def limited(request):
        return httpx.Response(429, text='{"error":"rate limit"}')

    caplog.set_level(logging.WARNING, logger="factcheck.llm")
    with pytest.raises(httpx.HTTPStatusError):
        _complete(_client(limited))
    assert "chutes failure" in caplog.text
    assert "status=429" in caplog.text
    assert "chutes balance/quota" not in caplog.text

    caplog.clear()

    def quota(request):
        return httpx.Response(429, text='{"error":"quota exceeded"}')

    with pytest.raises(httpx.HTTPStatusError):
        _complete(_client(quota))
    assert "chutes balance/quota" in caplog.text
    assert "status=429" in caplog.text


def test_bad_json_is_logged_with_the_model(caplog):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "Qwen/Qwen3.5-397B-A17B-TEE",
                "choices": [{"message": {"content": "not json"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    caplog.set_level(logging.WARNING, logger="factcheck.llm")
    result = _complete(_client(handler))
    assert result.content == "not json"
    assert "kind=bad_json" in caplog.text
    assert "Qwen/Qwen3.5-397B-A17B-TEE" in caplog.text
    assert "sk-live-secret" not in caplog.text
