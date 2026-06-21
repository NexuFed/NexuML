"""Automatic CUDA batch-size resolution helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from tqdm.auto import tqdm

from nexuml.core.types import AutoBatchSizeSpec

logger = logging.getLogger(__name__)


@dataclass
class BatchProbeAttempt:
    """Result of a single batch-size probe attempt."""

    batch_size: int
    status: str
    error: str | None = None
    seconds: float | None = None
    memory_peak_bytes: int | None = None
    memory_total_bytes: int | None = None
    memory_fraction: float | None = None


@dataclass
class BatchProbeResult:
    """Aggregated result of an automatic batch-size probe run."""

    selected_batch_size: int
    attempts: list[BatchProbeAttempt] = field(default_factory=list)
    safety: str = "largest"
    device: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_batch_size": self.selected_batch_size,
            "attempts": [attempt.__dict__ for attempt in self.attempts],
            "safety": self.safety,
            "device": self.device,
        }


def generate_candidates(config: AutoBatchSizeSpec) -> list[int]:
    """Generate bounded power-of-two candidates in ascending order.

    Returns:
        Sorted list of candidate batch sizes.

    Raises:
        ValueError: If the candidate strategy is not ``power_of_two``.
    """
    if config.candidates != "power_of_two":  # defensive; pydantic constrains this
        raise ValueError(f"Unsupported auto batch candidate strategy: {config.candidates}")
    value = 1
    while value < config.min:
        value *= 2
    candidates: list[int] = []
    while value <= config.max:
        candidates.append(value)
        value *= 2
    if config.min not in candidates:
        candidates.insert(0, config.min)
    return sorted(set(candidates))


def select_batch_size(successes: list[int], config: AutoBatchSizeSpec) -> int:
    """Select the effective batch size from successful candidates.

    Returns:
        The selected batch size.

    Raises:
        ValueError: If ``successes`` is empty or the safety policy is unsupported.
    """
    if not successes:
        raise ValueError("Cannot select an automatic batch size without successful candidates")
    largest = max(successes)
    if config.safety == "largest":
        return largest
    if config.safety == "previous_power_of_two":
        lower = [candidate for candidate in successes if candidate < largest]
        return max(lower) if lower else largest
    if config.safety == "margin":
        return largest
    raise ValueError(f"Unsupported auto batch safety policy: {config.safety}")


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True if the exception is a CUDA out-of-memory error."""
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def cuda_device_info() -> dict[str, Any]:
    """Return CUDA device metadata, or ``{"available": False}`` if no GPU is present."""
    if not torch.cuda.is_available():
        return {"available": False}
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "available": True,
        "index": device,
        "name": torch.cuda.get_device_name(device),
        "memory_free_bytes": free,
        "memory_total_bytes": total,
    }


def _cuda_memory_snapshot() -> tuple[int | None, int | None, float | None]:
    if not torch.cuda.is_available():
        return None, None, None
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    _free, total = torch.cuda.mem_get_info(device)
    total = int(total)
    fraction = peak / total if total else None
    return peak, total, fraction


def resolve_with_probe(
    config: AutoBatchSizeSpec,
    probe: Callable[[int], None],
) -> BatchProbeResult:
    """Try candidate batch sizes and return selected size plus observable metadata.

    Returns:
        Aggregated probe result with selected batch size.

    Raises:
        RuntimeError: If no candidate succeeds because candidates hit CUDA OOM or exceed
            the configured margin policy.
    """
    attempts: list[BatchProbeAttempt] = []
    successes: list[int] = []
    for candidate in tqdm(
        generate_candidates(config),
        desc="Batch size",
        unit="candidate",
    ):
        started = time.perf_counter()
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            probe(candidate)
            peak, total, fraction = _cuda_memory_snapshot()
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            attempts.append(
                BatchProbeAttempt(
                    batch_size=candidate,
                    status="cuda_oom",
                    error=str(exc),
                    seconds=time.perf_counter() - started,
                )
            )
            continue
        status = "success"
        if config.safety == "margin" and fraction is not None and fraction > config.margin:
            status = "over_margin"
        attempts.append(
            BatchProbeAttempt(
                batch_size=candidate,
                status=status,
                seconds=time.perf_counter() - started,
                memory_peak_bytes=peak,
                memory_total_bytes=total,
                memory_fraction=fraction,
            )
        )
        if status == "success":
            successes.append(candidate)

    if not successes:
        failed = ", ".join(str(attempt.batch_size) for attempt in attempts)
        raise RuntimeError(
            "Automatic batch-size probe failed for all candidates "
            f"in bounds [{config.min}, {config.max}]. Failed candidates: {failed}"
        )
    else:
        logger.info(
            f"Automatic batch-size probe found {len(successes)} eligible candidates "
            f"in bounds [{config.min}, {config.max}]. "
            f"Selected: {select_batch_size(successes, config)}"
        )

    return BatchProbeResult(
        selected_batch_size=select_batch_size(successes, config),
        attempts=attempts,
        safety=config.safety,
        device=cuda_device_info(),
    )
