# Physical policy meaning

Physical AI uses the existing NexuML lifecycle, not a parallel robotics framework.

| Foundation contract | Existing owner |
| --- | --- |
| Task | `EnvironmentDefinition`, `InteractionSpec`, `InteractionContract` |
| Experience | `DataSourceDefinition`, `DataSpec`, `NexuDataset` and existing loaders |
| Learning | Training configuration, algorithm definition and `NexuSession` |
| Policy artifact | `PolicySpec`, `CompiledPolicy` and complete-policy packaging |
| Evaluation | Existing episode evaluator and evaluation configuration; broader offline-policy evaluation remains separate work |

## Explicit physical metadata

`TensorFieldContract.physical` optionally carries a version-1
`PhysicalTensorContract`: units, reference frame, ordered joints, control mode,
scale/offset and sensor identity. The ordinary tensor shape/dtype/bounds and
`rate_hz` remain on `TensorFieldContract`.

Physical control metadata describes what the native adapter does; it does **not**
apply a second conversion. For example, SO-101's five normalized actions remain
bounded by `[-1, 1]`, while the native position controller maps them to joint
targets with `0.5 * action + 0` radians:

```python
from nexuml.core.types import PhysicalTensorContract, TensorFieldContract

action = TensorFieldContract(
    shape=(5,), dtype="float32", low=-1, high=1, rate_hz=30,
    physical=PhysicalTensorContract(
        units=("rad",),
        frame="robot_joint",
        joints=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
        control_mode="absolute_position",
        scale=(0.5,), offset=(0.0,),
    ),
)
```

Units, scales and offsets contain one broadcast value or one per final feature
axis. Joint order, when present, covers that axis. Observation layouts may
repeat joints across position/velocity/goal channels; an action's controlled
joints must be unique. A control mode requires explicit joints and scale;
zero/nonfinite scales, malformed widths and contradictory action/control-period
rates are rejected. This slice supports declared effort, absolute-position and
relative-position interpretations, not every possible controller.

`InteractionContract.physical` optionally carries a version-1
`PhysicalSystemContract`: embodiment, asset source/content identity and
calibration identity. IDs are compared exactly, not inferred from tensor width
or a robot name. The current SO-101 joint-reaching reference uses a content ID
derived from its fifteen declared raw authored asset-layer hashes. Its native
provider verifies those bytes and the actual joint/actuator properties before
policy actions. Calibration identifies the **simulated USD-authored joint zero**,
not measured real-arm calibration or permission to actuate hardware.

Only the inspected SO-101 reference populates the new metadata currently. Other
references retain their earlier contracts until their physical identities are
independently audited; this is not full embodiment coverage.

## Compatibility and legacy artifacts

Definitions and contracts serialize through the existing registry/JSON/YAML
paths and complete-policy package. Discovery and portable inference need no
simulator, reader, robotics backend or model download.

Before allocating an evaluation environment, the existing evaluator rejects
changed declared system identity, units, frames, joint order, control mode,
scale/offset, sensors, bounds or control rate. A matching shape is insufficient.
New policies with physical metadata also reject an environment that omits it.
Do not use `model_copy(update=...)` to bypass definition validation in user code.

Old contracts without physical metadata keep their previous shape, dtype and
action-description compatibility checks. They are not silently upgraded into
physically qualified artifacts. Absence is not a claim that hardware, calibration
or sensor mapping is correct. The historical EA SO-101 definition retains its
original identity; executing it still requires its original source profile.

Temporal axes, observation/action chunks, privileged inputs, learned processors,
caller-owned inference state and non-RL policy-evaluation configuration are
separate extensions. No hidden state, resampling, unit conversion, calibration
measurement or deployment safety layer is introduced by these metadata models.
