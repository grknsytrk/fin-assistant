"""Shared exceptions for KAP fetcher modules."""


class KapPacingDeadlineExceeded(RuntimeError):
    """Raised when waiting for the next KAP request slot would exceed a batch job deadline."""

