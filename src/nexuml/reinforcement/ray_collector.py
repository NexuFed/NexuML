"""Lazy Ray-specific teardown repair for the pinned TorchRL collector."""

from importlib.metadata import version

import ray
import torch
from torchrl.collectors.distributed.ray import RayCollector


def _validate_synchronous_rollout(batch, *, version: int, frames: int, workers: int):
    if (
        batch is None
        or batch.ndim < 2
        or batch.shape[0] != workers
        or batch.numel() != frames
        or batch.names[-1] != "time"
    ):
        raise RuntimeError("Ray collectors returned an incomplete rollout batch")
    # The root field can contain the prior observation's version at a batch boundary.
    received = batch.get(("next", "policy_version"), None)
    if (
        not isinstance(received, torch.Tensor)
        or received.dtype != torch.long
        or received.shape != batch.batch_size
        or not (received == version).all()
    ):
        raise RuntimeError(
            f"Ray rollout contains a missing or stale policy version, expected {version}"
        )


class _GracefulRayCollector(RayCollector):
    """Retain TorchRL collection/sync and validate rollout version before consumption."""

    _close_error: Exception | None = None

    def __init__(self, *args, **kwargs):
        if version("torchrl").split("+")[0] != "0.14.0":
            raise RuntimeError(
                "Ray collection's lifecycle adapter is qualified for TorchRL 0.14.0; "
                "use that version or requalify worker shutdown before enabling another release"
            )
        kwargs["collector_kwargs"] = {
            **kwargs.get("collector_kwargs", {}),
            "track_policy_version": True,
        }
        self._published_version = 0
        super().__init__(*args, **kwargs)

    def _weight_update_impl(self, policy_or_weights=None, *, worker_ids=None, **kwargs):
        if worker_ids is not None:
            raise ValueError("Synchronous RL requires updating every Ray rollout worker")
        super()._weight_update_impl(policy_or_weights, **kwargs)
        self._published_version += 1

    def iterator(self):
        for batch in super().iterator():
            _validate_synchronous_rollout(
                batch,
                version=self._published_version,
                frames=self.frames_per_batch,
                workers=self.num_collectors,
            )
            yield batch

    def stop_remote_collectors(self):
        if not self.remote_collectors:
            return  # Repeated teardown must not auto-initialize a new Ray runtime.
        try:
            # TorchRL 0.14 otherwise kills actors, also on iterator exhaustion, without close.
            closing = [worker.shutdown.remote() for worker in self.remote_collectors]
            _, pending = ray.wait(closing, num_returns=len(closing), timeout=30)
            if pending:
                raise TimeoutError("Ray worker close deadline exceeded")
            ray.get(closing)
        except Exception as error:
            # Let TorchRL release the weight-sync actor and transport before reporting failure.
            self._close_error = error
        finally:
            super().stop_remote_collectors()

    def shutdown(self, timeout=None, shutdown_ray=False):
        try:
            super().shutdown(timeout=timeout, shutdown_ray=shutdown_ray)
            if self._close_error is not None:
                error, self._close_error = self._close_error, None
                raise error
        finally:
            # A failed/timed-out remote close must not leak a collector-owned runtime lease.
            self._runtime_lease.release()
