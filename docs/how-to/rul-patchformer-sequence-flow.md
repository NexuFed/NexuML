# NEX-233 — Full-sequence Flow Matching in PatchFormer latent space

The previous PatchFormer flow experiments mean-pooled its DPAN output from
`[B,30,128]` to `[B,128]`, losing token-level temporal structure before
training the vector field. The new **SequenceFlow** component directly
learns velocities in `[B,30,128]`, with no temporal pooling or flattening
inside the flow model.

## Model architecture

Three registered `SequenceTimeConditionedVectorField` backbones share
the same public `SequenceFlow` interface and native Euler integration:

| Backbone | Temporal inductive bias | Implementation |
|---|---|---|
| Transformer | Global content-based attention | Sinusoidal sequence positions, two pre-LN encoder layers, 4 attention heads |
| Conformer | Global attention plus local convolution | Two macaron FFN / attention / depthwise-GLU convolution blocks, kernel 7 |
| CfC | Sequential recurrent continuous-time gating | Two fully connected CfC recurrent layers, closed-form gating between candidate states |

All use hidden width 128 and output a per-token 128-dimensional velocity.
The CfC recurrent layers implement the **default CfC gated
interpolation** using learned time coefficients and a fixed elapsed
token interval. They are not LTC ODE integration. This CfC state
evolution is distinct from the outer flow-time Euler integration.

Each vector field uses the normal NexuML Flow Matching path:

```text
C-MAPSS sensor history [B,30,14/15]
                 |
         frozen PatchFormer DPAN
                 |
      full embedding [B,30,128]
                 |
     train-only feature standardization
                 |
      healthy / degraded endpoint pairs
                 |
        PairedLinearFlowPath
        z_t=(1-t)z_0+t*z_1
                 |
         SequenceFlow field
       Transformer / Conformer / CfC
       v_theta(z_t,t) [B,30,128]
                 |
           FlowMatchingLoss
         MSE(v_theta, z_1-z_0)
                 |
      Euler transport on [B,30,128]
                 |
     validation sliced-Wasserstein
```

The RUL predictor/FAN head of PatchFormer is **not called** while
preparing these endpoints. The pretrained DPAN is frozen. We preserve
per-time-step DPAN tokens and position-specific standardization
statistics fitted on training endpoints only.

SWD flattens each **completed sequence** to a vector for the
distribution comparison only; it does not flatten sequence features
before the vector field or its Euler integrator.

The old mean-pooled flow scenarios and their stored artifacts remain
unchanged and independently reproducible.

## Run three matched FD001 seed-42 experiments

All experiments use the same fixed pretrained PatchFormer seed, same
asset-disjoint C-MAPSS train/val splits and randomly paired healthy
(RUL >= 100) / degraded (RUL <= 30) endpoints, 512 train pairs,
128 validation pairs, 250 training epochs maximum, 50-step Euler
integration, and validation-SWD checkpoint selection.

Train the relevant upstream controlled PatchFormer first if missing:

```bash
nexuml resolve maintenance-rul-controlled-fd001-patchformer-s42 \
  --output .experiments/rul-patchformer-fd001-s42.yaml
nexuml train --config .experiments/rul-patchformer-fd001-s42.yaml
```

Then, using only native NexuML resolve/train commands:

```bash
git pull
mkdir -p .experiments/rul/sequential

for architecture in transformer conformer cfc; do
  name="maintenance-rul-patchformer-sequence-$architecture-fd001-s42"
  config=".experiments/rul/sequential/$name.yaml"
  nexuml resolve "$name" --output "$config" || exit 1
  nexuml train --config "$config" || exit 1
done
```

Trained checkpoints are written under
`.experiments/rul/checkpoints/rul-FD001-patchformer-sequence-<architecture>-flow-seed42/`.
Each experiment's distinct full-sequence endpoint artifact is saved
under `.experiments/rul/prepared/`.

Compare:
- `val/flow_matching_loss` for velocity regression accuracy
- `val/healthy_to_faulty_swd` for the baseline endpoint gap
- `val/transport_swd` and `val/swd_relative_reduction` for transport
  alignment after 50 Euler steps

**Flow-time `t` is the interpolation variable in [0,1], not engine
chronological age or remaining life.** This is an unconditional
distributional transport comparison, not yet a standalone RUL benchmark.
Downstream RUL prediction using sequential flow states requires its
own readout experiment; the previous scalar 128-D flow checkpoint
loader and RUL heads are intentionally unchanged.

## Downstream full-sequence RUL evaluation

The matched **Transformer, Conformer and CfC flow-only RUL predictors**
and their encoder-only controls are now implemented. They freeze the
upstream DPAN and Flow, preserve all 30 temporal tokens, and train the
same attention-weighted RUL head with and without flow displacement.

Run all six FD001 seed-42 RUL experiments and generate a paired
100-engine evaluation report with:

```bash
git pull
bash scripts/launch_patchformer_sequence_rul_fd001_s42.sh
```

See [full-sequence Flow RUL evaluation](rul-sequence-flow-rul.md) for
the model contracts, the 36 registered scenarios, and the held-out
comparison procedure.

## FD003 and multiple seeds

Scenarios are registered for FD001 and FD003, each with seeds 42, 43,
and 44, for all three backbones (**18 configurations** in total).
For example:

```text
maintenance-rul-patchformer-sequence-transformer-fd001-s42
maintenance-rul-patchformer-sequence-conformer-fd001-s42
maintenance-rul-patchformer-sequence-cfc-fd001-s42
maintenance-rul-patchformer-sequence-transformer-fd003-s43
maintenance-rul-patchformer-sequence-conformer-fd003-s44
maintenance-rul-patchformer-sequence-cfc-fd003-s42
```

Do not select model designs using FD003 official test RUL labels.

## Implementation

- `layers/generative/flow/sequence_vector_field.py` — three sequence
  vector-field backbones
- `layers/model/flow.py` — `SequenceFlow` wrapper, native `_FlowRuntime`
  and Euler integrator reuse
- `data/maintenance/patchformer_flow.py` — optional full-sequence DPAN
  endpoint preparation without FAN/RUL
- `data/maintenance/prepared_endpoints.py` — strict `[N,2,L,D]` shape
  and provenance validation
- `evaluation/maintenance/ablation_validation.py` — sequence-aware
  transport SWD
- `scenarios/maintenance/rul_patchformer_sequence_flow.py` — 18
  matched registered experiments
- `tests/library/test_sequence_flow.py` — sequence gradients, integration,
  SWD, native resolve/compile, and full-DPAN endpoint tests

## Verification

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_sequence_flow.py \
       tests/library/test_flow_matching.py \
       tests/library/test_rul_patchformer_study.py \
       tests/library/test_rul_advanced_native_cli.py \
       tests/_registry -v
```

These are normal NexuML components. No base structure changes, global
environment variables or manually configured model checkpoints are
required. Repository changes are pushed, but tests and the experiments
must still be executed in an installed NexuML environment.
