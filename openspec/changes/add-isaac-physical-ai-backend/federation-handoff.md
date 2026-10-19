# NexuML → NexuFL SO-101 learning contract (planned)

**Owner:** NexuML `feature/nex-212-add-reinforcement-learning` → tasks §0 / F01. **Consumer:** NexuFL `unify-client-initiated-federation-and-library` → `openspec/changes/federate-so101-imitation-learning/`. This file specifies a target handoff; it is **not** an implemented API.

## Minimal local proof and PR boundary
Native SO-101 reaching → genuine expert episodes (not invented labels) → small state-only behavior cloning with NexuML's existing trainer → clean policy export/reload → held-out **closed-loop simulator task-success** evidence (expert, untrained and learned reported separately). Then test a callable bounded **Python SDK** fit that takes new externally provided compatible tensor weights for each call and returns the changed named tensors, eligible training-window count, episode count and approved metrics. Stop for an early **narrow PR**, avoiding the complete Physical AI branch/roadmap dependency.

The SDK surface below defines **behavior**, not a second trainer or a mandatory exact function name:

```python
result = local_fit(
    trusted_recipe=operator_installed_recipe,
    initial_tensors=approved_model_state,
    local_data=private_approved_episode_dataset,
    budget={"steps": 50},
    seed=42,
    optimizer_mode="reset",
)
# updated_tensors + approved_scalars + train_windows + episodes + provenance
```

Use the current NexuML compilation, `NexuSession`, Lightning training, validation and export semantics. A Python client can run without any REST server; **no REST API** in this milestone. Do not add `FederatedTrainer`, a second model graph, NexuFL SDK import or gRPC transport to NexuML.

## Interoperability contract to freeze before PR
- **Trusted code:** fixed locally installed model/recipe/package (allowlisted, content-hash pinned). Never accept arbitrary server-selected Python import paths, remote source/pickle or execution-bearing `torch.package` over the federated wire.
- **Portable weights:** exact named trainable tensors, dtypes/shapes, architecture/recipe identity, content/base version and explicit `full` vs `delta` update. NexuFL will use its own Safetensors/model-manifest, checked artifact transfer and mTLS.
- **Physical semantics:** SO-101 asset/revision; actor-visible state/goal keys; five controlled joints / fixed gripper as appropriate; ordering, units, scaling, bounds, frame, command interval, processor/normalization identity; do not infer compatibility from matching dimensions.
- **Local data:** real episodic demonstrations, timestamps/action windows, traceable original sources and site-ready disjoint partitions. Use train-only normalization; record eligible windows and distinct episodes (not optimizer iterations or padded action positions) for reproducible sample-weighted FedAvg. Never expose raw trajectory/video/private local filesystem paths to the server.
- **Learning state:** first reference is stateless across FL rounds with explicit **optimizer reset per round**; client-private continuation requires a later explicit version/state-owner receipt and fresh tests. Fixed seeds/budgets/trainable scopes and bounded, finite metrics. Repeat fit with different initial weights must work.
- **Artifact split:** local trusted model implementation vs shareable portable numeric model tensors vs private site-only data/optimizer state. Inference package includes required preprocessors/action transforms; no simulator handle/training critic in the exported actor.
- **Preflight/failure:** reject mismatched joint/action/processor contracts, tensor keys/shape/dtypes/NaNs, unauthorized dataset handles, unsupported state requests or remote executable assets before training.

## Acceptance and handoff
Evidence must include an exact tested pinned Isaac/SO-101 task and expert dataset, learned closed-loop evaluation after clean export, a genuine second external-initial-weight fit, regression and negative tests, SDK usage, artifact/contract fixtures, seed/episode counts and commit/wheel identity. Preserve source hashes and observed failures. Do not claim network, federation, GPU Kubernetes or physical robot success.

**At task 0.8 stop and open/prepare a narrow NexuML implementation PR** against an appropriate stable integration base. The existing large feature branch need not be merged wholesale. Only after a reviewed pinned NexuML artifact is available should NexuFL independently implement local four-client FedAvg, actual outbound gRPC, Kubernetes parallelism, visual ACT and personalized/active-learning follow-ups.
