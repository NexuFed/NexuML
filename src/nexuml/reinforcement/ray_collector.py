"""Synchronous Ray policy publication and teardown for the pinned TorchRL collector."""

from importlib.metadata import version

import ray
import torch
from tensordict import TensorDictBase
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

    def __init__(self, *args, admission_timeout_s=600, **kwargs):
        if version("torchrl").split("+")[0] != "0.14.0":
            raise RuntimeError(
                "Ray collection's lifecycle adapter is qualified for TorchRL 0.14.0; "
                "use that version or requalify worker shutdown before enabling another release"
            )
        kwargs["collector_kwargs"] = {
            **kwargs.get("collector_kwargs", {}),
            "track_policy_version": True,
        }
        # The default Gloo scheme starts an extra actor outside the prepared rollout runtime.
        kwargs["weight_sync_schemes"] = {}
        self._published_version = 0
        self._admission_timeout_s = admission_timeout_s
        self._unadmitted_refs = {}
        try:
            super().__init__(*args, **kwargs)
        except BaseException as error:
            if hasattr(self, "_remote_collectors"):
                try:
                    # Unscheduled actors cannot acknowledge graceful shutdown.
                    for worker in self._unadmitted_refs.values():
                        ray.kill(worker, no_restart=True)
                        self.remote_collectors.remove(worker)
                    self.shutdown()
                except Exception as cleanup:
                    error.add_note(f"Ray collector startup cleanup also failed: {cleanup}")
            raise

    def add_collectors(self, *args, **kwargs):
        super().add_collectors(*args, **kwargs)
        self._unadmitted_refs = {worker.stats.remote(): worker for worker in self.remote_collectors}
        if self._unadmitted_refs:
            ready, pending = ray.wait(
                list(self._unadmitted_refs),
                num_returns=len(self._unadmitted_refs),
                timeout=self._admission_timeout_s,
            )
            for ref in ready:
                self._unadmitted_refs.pop(ref)
            ray.get(ready)
            if pending:
                raise TimeoutError(
                    f"Ray rollout admission timed out after {self._admission_timeout_s:g} s: "
                    f"{len(pending)}/{len(self.remote_collectors)} collector(s) did not start. "
                    "Check the requested GPU/CPU resources, selectors and prepared worker runtime."
                )

    def _make_collector(self, cls, *, env_maker, policy, other_params):
        # Ray checks actor truthiness when reporting errors; TorchRL's infinite budget breaks len().
        batches = (self.total_frames + self.frames_per_batch - 1) // self.frames_per_batch
        if policy is not None:
            other_params["policy"] = policy
        return cls(
            env_maker,
            total_frames=batches * other_params["frames_per_batch"],
            **other_params,
        )

    def _weight_update_impl(
        self, policy_or_weights=None, *, worker_ids=None, model_id=None, weights_dict=None, **kwargs
    ):
        if worker_ids is not None:
            raise ValueError("Synchronous RL requires updating every Ray rollout worker")
        if model_id not in (None, "policy") or weights_dict is not None or kwargs:
            raise ValueError("Synchronous RL publishes only the rollout policy")
        if not isinstance(policy_or_weights, TensorDictBase):
            raise TypeError("Synchronous RL requires a complete policy TensorDict")
        weights = ray.put(policy_or_weights.detach().cpu())
        acknowledgements = [
            worker.update_policy_weights_.remote(weights) for worker in self.remote_collectors
        ]
        ray.get(acknowledgements, timeout=self._admission_timeout_s)
        self._weight_sync_initialized = True
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
