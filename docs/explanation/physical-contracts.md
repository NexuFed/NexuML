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

Temporal execution, action-chunk scheduling, privileged inputs, learned processors,
caller-owned inference state and non-RL policy-evaluation configuration are
separate extensions. No hidden state, resampling, unit conversion, calibration
measurement or deployment safety layer is introduced by these metadata models.

## Typed sequence and multimodal compile inputs

`TensorFieldContract.axes` names fixed **feature** dimensions, excluding the
TensorDict's inference/environment batch dimensions. Names are unique and match
the tensor rank; reserved `batch`/`environment` names cannot disguise an outer
batch as a feature axis. For example, a sample with shape `(2, 3, 5)` and axes
`("agent", "time", "joint")` retains all three dimensions when a batch of samples
is compiled or evaluated. Names do not imply a recurrent executor or an alignment
policy.

For offline graphs, `DataSpec.input_contracts` supplies the same typed input
metadata used by interaction compilation. This is a compile-metadata fragment;
attach a real source/loader and compatible pipeline for training:

```python
from nexuml.core.types import DataSpec, TensorFieldContract

data = DataSpec(
    feature_key="state",
    input_contracts={
        "state": TensorFieldContract(
            shape=(2, 3, 5), dtype="float32", axes=("agent", "time", "joint"),
        ),
        "tokens": TensorFieldContract(
            shape=(2, 16), dtype="int64", axes=("time", "token"),
        ),
        "mask": TensorFieldContract(
            shape=(2, 16), dtype="bool", axes=("time", "token"),
        ),
        "images": TensorFieldContract(
            shape=(2, 3, 64, 64), dtype="uint8",
            axes=("time", "channel", "height", "width"),
        ),
    },
)
```

Legacy `input_shapes` and source `feature_shape` inference still work. Typed and
legacy shapes may coexist, but overlapping declarations must agree. Typed
inputs carry dtype, axes and optional physical metadata through the same
`LayerBuildContext.metadata["input_metadata"]` seam. Unsupported tensor dtypes
fail before layer construction. The first loaded sample is checked against the
declared contracts before dataset-backed training compiles.

Shape propagation preserves supported integer/token/mask dtypes through intermediate
outputs, so an integer identity layer followed by an embedding does not receive
a fabricated floating-point dummy. It infers output shapes/dtypes, **not** new
physical identities or named axes for an arbitrary transform. Layers retain
responsibility for semantic transformations and explicit output annotations.
Feature time/agent dimensions are not automatically flattened or resampled.
Specialized intermediate dtypes outside the declared input family, such as
quantized/float8 tensors, retain the prior shape-only dummy fallback; they are
not advertised as fully dtype-qualified compile inputs.

Package I/O metadata and ONNX dummy inputs use these same declared shapes and
dtypes. This does not promise every model/operator/backend can export to ONNX,
or make fixed time dimensions dynamically sized. New policies with declared axes
reject mismatched or missing axes in evaluation; legacy absence keeps the old
compatibility rules.
