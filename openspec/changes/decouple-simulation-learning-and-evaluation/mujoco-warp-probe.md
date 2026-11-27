# Task 5.1 — MuJoCo Warp API and CUDA tensor probe

Choose direct upstream MuJoCo Warp as the smallest candidate for a future
Torch-native adapter. Keep Gymnasium MuJoCo unchanged. This route uses the
prepared environment's existing native packages and tested Cartpole fixture
asset. It does not need JAX or MuJoCo Playground, neither of which is installed
in this profile. The separate Playground, locomotion and camera requirements
in `add-isaac-physical-ai-backend` still need their own pinned compatibility
and dependency-composition evidence. No new image or dependency was added.

## What ran

Python 3.12.16, MuJoCo and MuJoCo Warp 3.12.0, Warp 1.17.0, PyTorch
2.11.0+cu130, RTX 4080 SUPER `cuda:0`, driver 610.43.03. Used the same
MuJoCo XML as the paired CPU reference, SHA-256
`d9657349921533445007a0f83eac74d2ba274ba0cc49c585618d00bb98b75704`.
The GPU probe ran in a fresh process without Isaac imports.

1. Load the XML with `mujoco.MjModel.from_xml_path`. Solve the CPU reference
   from a fresh `MjData` with each of `(+0.5, -0.5, 0.0)` for two steps.
2. On `cuda:0`, use `mujoco_warp.put_model` and `make_data(nworld=3)`.
   Run `mujoco_warp.step` twice on a shared
   `warp.ScopedStream(warp.stream_from_torch(torch.cuda.Stream(...)))`.
   `warp.to_torch` exposed float32 CUDA aliases with matching pointers for
   controls, positions, velocities and per-world clocks.
3. Use `warp.from_torch` on the CUDA mask `[false, true, false]` without a
   copy. `mujoco_warp.reset_data(..., reset=mask)` cleared only that world's
   state, input and time. Everything checked in the other rows was bitwise
   unchanged.

CUDA cart velocities at the native 60-Hz boundary were `+0.069356859`,
`-0.069515556` and `-0.000079413 m/s`. Position and velocity matched CPU
for these two steps (rtol `1e-4`, atol `1e-6`). The initial exact-zero and
pure-sign-symmetry assumptions were wrong: CPU MuJoCo reproduced the
asset's small drift and asymmetry. Do not assert general CPU/GPU
trajectory equivalence. After reset, clocks were approximately
`[1/60, 0, 1/60] s`.

The corrected probe exited 0. Raw receipts and initial diagnostic:
`/tmp/opencode/nexuml-decouple-mjw-substrate-rZwnaS/`.
`probe.json` SHA-256:
`417516e049afc351b4c41467a929b49918964c6c2227ce6b30c21daf0f5565bf`.

This qualifies only the API substrate for 5.1. There is no GPU NexuML
environment, rollout/PPO, in-process Isaac/MuJoCo interoperability,
rendering, throughput or distributed support claim. The broader learning
and placement gate remains unchanged.
