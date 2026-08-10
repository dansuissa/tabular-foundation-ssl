"""Shared GPU timing / memory helpers for foundation-model wrappers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class GPUTimingMeta:
    cold_load_seconds: float | None = None
    warm_inference_seconds: float | None = None
    peak_gpu_memory_mb: float | None = None
    device: str | None = None
    cuda_available: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def resolve_torch_device(requested: str | None = None) -> str:
    try:
        import torch
    except ImportError as exc:
        from src.exceptions import OptionalDependencyError

        raise OptionalDependencyError("torch", "Install torch in ssl-tfm.") from exc

    if requested is None or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def reset_peak_memory(device: str) -> None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        return


def peak_memory_mb(device: str) -> float | None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            return float(torch.cuda.max_memory_allocated() / (1024**2))
    except Exception:
        return None
    return None


@contextmanager
def timed_section() -> Iterator[list[float]]:
    box: list[float] = []
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box.append(time.perf_counter() - t0)


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or name in {"cudaoutofmemoryerror", "outofmemoryerror"}
    )
