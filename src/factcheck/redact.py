"""Keep API keys out of process logs. httpx otherwise prints the full request URL."""

from __future__ import annotations

import logging
import re

_QUERY = re.compile(r"(?i)\b(key|api_key|token)=([^&\s\"']+)")
_AUTHORIZATION = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?([^\s\"']+)"
)

_INSTALLED = False


def redact_secrets(text: str) -> str:
    """Replace secret query values and Authorization values. The rest of the line stays."""
    cleaned = _QUERY.sub(r"\1=REDACTED", text)
    return _AUTHORIZATION.sub(r"\1REDACTED", cleaned)


def install_log_redaction() -> None:
    """httpx and httpcore stay at WARNING. Every log record is redacted."""
    global _INSTALLED
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if _INSTALLED:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        original = record.getMessage

        def get_message() -> str:
            return redact_secrets(original())

        record.getMessage = get_message
        return record

    logging.setLogRecordFactory(factory)
    _INSTALLED = True
