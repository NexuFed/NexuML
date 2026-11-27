# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/isaac-sim:6.2.0@sha256:bcb96e0b87c28ad446a2a6a2f0fdb7d82b9867f18b96a94c796912c61f07337f

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Keep the vendor Kit interpreter and graphics stack; add binary system tools only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libsndfile1 \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Prepared inputs contain existing binary artifacts, never source-build fallbacks.
COPY uv /usr/local/bin/uv
COPY upstream/install_carb_env_shim.sh /opt/nexuml-build/install_carb_env_shim.sh
RUN bash /opt/nexuml-build/install_carb_env_shim.sh /isaac-sim

ENV VIRTUAL_ENV=/env \
    UV_PROJECT_ENVIRONMENT=/env \
    UV_PYTHON=/env/bin/python \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/env/bin:$PATH" \
    PYTHONPATH=/env/lib/python3.12/site-packages \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
    ISAAC_PATH=/isaac-sim \
    CARB_APP_PATH=/isaac-sim/kit \
    EXP_PATH=/isaac-sim/apps \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMNI_CRASHREPORTER_ENABLED=0 \
    OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD=1

RUN uv --no-config venv --python /isaac-sim/kit/python/bin/python3 /env

# Preserve genuine VCS provenance while reusing the previously built Lab wheel.
# A missing cached binary fails under --offline/--no-build; it cannot build Lab.
RUN --mount=type=bind,source=uv-lab-cache,target=/tmp/uv-lab-cache,rw \
    uv --no-config --offline --cache-dir /tmp/uv-lab-cache pip install \
        --python /env/bin/python --no-build --no-deps --link-mode copy \
        'isaaclab @ git+https://github.com/isaac-sim/IsaacLab.git@70ab24fec3eabf51222817db0cdb5a4e7133bd7e#subdirectory=tools/wheel_builder'

COPY external-requirements.txt /opt/nexuml-build/external-requirements.txt
COPY overrides.txt /opt/nexuml-build/overrides.txt
COPY workspace-wheels/antlr4_python3_runtime-4.9.3-py3-none-any.whl /opt/nexuml-build/wheels/
COPY antlr-requirement.txt /opt/nexuml-build/antlr-requirement.txt

# The full overlay was jointly resolved against fresh current NexuML wheels.
# Isaac Sim itself comes from the pinned base, not a duplicate pip installation.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=4 \
    uv --no-config pip install --python /env/bin/python \
        --require-hashes --only-binary :all: --torch-backend cu130 \
        --overrides /opt/nexuml-build/overrides.txt \
        --default-index https://pypi.org/simple \
        -r /opt/nexuml-build/external-requirements.txt \
        -r /opt/nexuml-build/antlr-requirement.txt

COPY workspace-wheels/nexuml*.whl /opt/nexuml-build/wheels/
COPY workspace-requirements.txt /opt/nexuml-build/workspace-requirements.txt
RUN uv --no-config pip install --python /env/bin/python \
    --offline --require-hashes --no-deps --only-binary :all: \
    -r /opt/nexuml-build/workspace-requirements.txt

COPY environment.sh /usr/local/bin/nexuml-isaac
RUN chmod 755 /usr/local/bin/nexuml-isaac

# Use upstream's own container setup to prevent bundled Torch/CUDA shadowing.
RUN ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES /usr/local/bin/nexuml-isaac /env/bin/python -c \
    'from isaaclab.cli.commands.install import _repoint_prebundle_packages; _repoint_prebundle_packages()'

# Inspect the pod's NetBird interface and sockets without running a VPN daemon here.
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

USER isaac-sim
WORKDIR /isaac-sim
HEALTHCHECK NONE
EXPOSE 49100/tcp 47998/udp

ENTRYPOINT ["/usr/local/bin/nexuml-isaac"]
CMD ["/bin/bash"]
