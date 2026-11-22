import logging

from factcheck.app import create_app
from factcheck.redact import redact_secrets

from tests.conftest import make_service


def test_request_urls_and_authorization_headers_are_redacted(caplog):
    service, _, _, settings = make_service()
    create_app(settings=settings, service=service)
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
    caplog.set_level(logging.INFO, logger="factcheck.testredact")
    secret = "AIzaSy-super-secret-google-key"
    token = "sk-live-authorization-token"
    logging.getLogger("factcheck.testredact").info(
        "HTTP Request: GET https://factchecktools.googleapis.com/v1alpha1/claims:search?key=%s&query=taxes",
        secret,
    )
    logging.getLogger("factcheck.testredact").info("Authorization: Bearer %s", token)
    logging.getLogger("factcheck.testredact").warning("api_key=abcd token=efgh")
    text = caplog.text
    assert secret not in text
    assert token not in text
    assert "abcd" not in text
    assert "efgh" not in text
    assert "key=REDACTED" in text
    assert "query=taxes" in text
    assert "Authorization: REDACTED" in text or "Authorization: Bearer REDACTED" in text
    assert redact_secrets(f"key={secret}") == "key=REDACTED"
