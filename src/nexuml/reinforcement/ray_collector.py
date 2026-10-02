"""Lazy Ray-specific teardown repair for the pinned TorchRL collector."""

import ray
from torchrl.collectors.distributed.ray import RayCollector


class _GracefulRayCollector(RayCollector):
    """Retain TorchRL collection/sync while closing environments before actor termination."""

    def stop_remote_collectors(self):
        if not self.remote_collectors:
            return  # Repeated teardown must not auto-initialize a new Ray runtime.
        try:
            # TorchRL 0.14 otherwise kills actors, also on iterator exhaustion, without close.
            # ponytail: 30s graceful-close bound; expose a deadline for slower runtimes if needed.
            ray.get([worker.shutdown.remote() for worker in self.remote_collectors], timeout=30)
        finally:
            super().stop_remote_collectors()

    def shutdown(self, timeout=None, shutdown_ray=False):
        try:
            super().shutdown(timeout=timeout, shutdown_ray=shutdown_ray)
        finally:
            # A failed/timed-out remote close must not leak a collector-owned runtime lease.
            self._runtime_lease.release()
