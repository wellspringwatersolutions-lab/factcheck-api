"""Fact-checking for agents and researchers."""

from factcheck.branding import SERVICE_NAME
from factcheck.redact import install_log_redaction

install_log_redaction()

__version__ = "0.1.0"
__all__ = ["SERVICE_NAME", "__version__"]
