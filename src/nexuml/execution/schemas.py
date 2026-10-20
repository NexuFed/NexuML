"""Safe, sourced resource snapshots; unknown is distinct from observed zero."""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class NodeResources(BaseModel):
    """Observable scheduling quantities for one node, without credentials."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str
    allocatable: dict[str, float] = Field(default_factory=dict)
    unallocated: dict[str, float] | None = None
    eligible: bool | None = None
    accelerator_type: str | None = None


class ResourceSnapshot(BaseModel):
    """Read-only advisory capacity in an explicit target scope."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    backend: str
    source: str
    target: str | None = None
    context: str | None = None
    namespace: str | None = None
    observed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    complete: bool = False
    reachable: bool | None = None
    api_supported: bool | None = None
    submission_allowed: bool | None = None
    controller_healthy: bool | None = None
    visible_nodes: int | None = None
    matching_nodes: int | None = None
    allocatable: dict[str, float] | None = None
    host_total: dict[str, float] | None = None
    units: dict[str, str] = Field(default_factory=lambda: {"CPU": "cores", "memory": "bytes"})
    unallocated: dict[str, float] | None = None
    quota_remaining: dict[str, float] | None = None
    nodes: list[NodeResources] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    namespaces: list[str] = Field(default_factory=list)
    ray_clusters: list[dict[str, str]] = Field(default_factory=list)
    roles: list[dict] = Field(default_factory=list)
    options: dict[str, list[str]] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)


class NativeReference(BaseModel):
    """Exact native identity for safe observation and cancellation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    backend: str
    version: str = "1"
    context: str
    namespace: str
    api_version: str
    kind: str
    name: str
    uid: str
