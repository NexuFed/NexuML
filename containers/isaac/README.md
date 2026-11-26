# Combined Isaac workstation image

The pinned NVIDIA Isaac Sim container provides Kit Python 3.12 and the full
editor. Headless NexuML training uses the same image. LeRobot is separate.
Build the image locally. No workflow publishes it. When authorized to share
it with the cluster, use its network-restricted Harbor registry. Do not mirror
the NVIDIA-derived image to public registries.

## Build locally

From the repository root, use Python 3.12 and the `uv` 0.13.5 Linux AMD64 binary
identified in `artifacts.json` (set `CONTAINER_UV` if needed). Set `ISAAC_ARTIFACTS`
to the approved local input directory: it contains `uv-lab-cache/` with the Lab
wheel, expanded cache entry and VCS metadata, and
`workspace-wheels/antlr4_python3_runtime-4.9.3-py3-none-any.whl`. Preserve the
relative symlinks when moving the Lab cache. An earlier prepared context with
these exact inputs can serve as the artifact directory.

```bash
CONTEXT=/tmp/opencode/nexuml-isaac-context
.venv/bin/python containers/isaac/stage.py "$CONTEXT" \
  --artifacts "${ISAAC_ARTIFACTS:?Set the approved local artifact directory}"
docker buildx build --platform linux/amd64 --load \
  --tag nexuml-isaac:workstation "$CONTEXT"
docker run --rm --network=none nexuml-isaac:workstation nexuml --help
```

The context directory must be new. Staging builds both NexuML wheels from the
checkout and verifies the approved artifact hashes. Docker installs the reviewed
binary-only pins without upgrading them. Lab's bundled cache preserves its real
VCS installation metadata without building Lab; a version-only registry wheel
is not equivalent. There is no fresh bootstrap without these Lab and ANTLR
artifacts. Do not install a second pip simulator.

## Refresh dependencies

Both `pyproject.toml` files define the application dependency ranges. The image
has its own `external-requirements.txt`: the simulator comes from the NVIDIA
base rather than the pip packages in the developer `uv.lock`. Running `uv sync`
in the image would apply the developer environment instead of preserving this
profile. In a developer environment, `uv sync --locked` uses the workspace lock,
and `uv lock --upgrade-package NAME` explicitly requests an update there.

```bash
.venv/bin/python containers/isaac/stage.py --check
.venv/bin/python containers/isaac/stage.py /tmp/opencode/nexuml-isaac-refreshed \
  --artifacts "${ISAAC_ARTIFACTS:?Set the approved local artifact directory}" \
  --refresh-lock --upgrade-package NAME
```

The check fails on changed dependency inputs, not on release version bumps.
`--refresh-lock` runs `uv pip compile` against the current wheels and Isaac
constraints, keeps compatible existing pins unless asked to upgrade, and writes
the updated hash lock and a build context. Use `--upgrade` instead to refresh all
unpinned dependencies. Neither mode changes the workspace venv or `uv.lock`.
The compatibility pins in `requirements.in` and `constraints.in` stay in force.
Review the lock diff and run the image smoke before using new pins. The resolver
can check version and wheel compatibility, not whether a GPU runtime works.

The one reviewed metadata exception is OvPhysX's `packaging<24` declaration with
the native profile's `packaging==25.0`. The effective graph must be checked and
that declared mismatch reported; do not advertise an ordinary clean `pip check`.
Virtualenv 21.7.8 is retained because newer virtualenv requires packaging 26.3.

## Launch modes

The default command is Bash. For the editor, explicitly run
`/isaac-sim/runheadless.sh --enable isaacsim.ros2.bridge` with NVIDIA GPU access,
graphics/video capabilities, explicit EULA acceptance and a secured WebRTC route.
The image includes `ip` and `ss` for diagnosing the Pod's NetBird interface.
The Kubernetes operator can inject NetBird as a sidecar. Optional telemetry
consent is not enabled. See the [Kubernetes templates](../../.k8s/isaac/README.md).

For training, replace the image command with the normal CLI:

```bash
docker run --rm --gpus '"device=0"' --shm-size=2g \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  nexuml-isaac:workstation nexuml train isaac-cartpole-ppo
```

That command illustrates process startup, not persistent training output: mount
your owned workspace and select durable logs/export destinations for real runs.
The native task still needs access to its Isaac assets, including the ground
plane; the workstation image does not bundle every task asset.
Inside an existing pod, use `/usr/local/bin/nexuml-isaac nexuml ...` so Kit's
bindings are initialized before the ordinary CLI. The wrapper only exports the
vendor environment and executes the command; it never installs dependencies.

Kit-backed training and the editor are separate native processes. Stop/save the
editor before reusing its single GPU for training, or allocate a separate training
pod from the same image and persistent assets. A USD scene still needs an Isaac
Lab task definition for observations, actions, rewards, resets and termination.
