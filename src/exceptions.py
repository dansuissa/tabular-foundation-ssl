"""Project-wide exceptions for optional deps, capability gaps, and protocol errors."""
from __future__ import annotations


class OptionalDependencyError(ImportError):
    """Required optional package is missing; record per-run unsupported status."""

    def __init__(self, package: str, hint: str | None = None) -> None:
        self.package = package
        msg = f"Optional dependency '{package}' is not available."
        if hint:
            msg = f"{msg} {hint}"
        super().__init__(msg)


class UnsupportedMethodError(RuntimeError):
    """Method cannot run for a precise, logged scientific reason."""

    def __init__(self, method: str, reason: str, status: str = "unsupported") -> None:
        self.method = method
        self.reason = reason
        self.status = status
        super().__init__(f"{method}: {reason}")


class TFMOOMError(RuntimeError):
    """Foundation-model GPU out-of-memory after retries."""

    def __init__(self, method: str, detail: str) -> None:
        self.method = method
        self.detail = detail
        super().__init__(f"{method} OOM: {detail}")


class LeakageError(RuntimeError):
    """Detected or attempted train/val/test protocol leakage."""
