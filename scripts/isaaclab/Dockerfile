FROM nvcr.io/nvidia/isaac-sim:6.1.0@sha256:0c16dd67d09a70ea474f2c809a13c4d09bd23c738184cf1bf28af487ecd39080

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/isaaclab/constraints.txt /opt/nexuml-isaac-constraints.txt
ENV ISAACLAB_PATH=/opt/isaaclab \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL="https://pypi.org/simple https://download.pytorch.org/whl/cu128" \
    PIP_CONSTRAINT=/opt/nexuml-isaac-constraints.txt \
    UV_CONSTRAINT=/opt/nexuml-isaac-constraints.txt \
    GIT_LFS_SKIP_SMUDGE=1 \
    HUB__ARGS__DETECT_ONLY=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN git clone --depth 1 --branch v3.0.0-EA https://github.com/isaac-sim/IsaacLab.git /opt/isaaclab \
    && test "$(git -C /opt/isaaclab rev-parse HEAD)" = ae37b028ea415c91ea2bc32609efcd759ed2b974 \
    && ln -s /isaac-sim /opt/isaaclab/_isaac_sim \
    && bash /opt/isaaclab/docker/scripts/install_carb_env_shim.sh /isaac-sim \
    && /opt/isaaclab/isaaclab.sh --install core

WORKDIR /workspace/nexuml
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY library/pyproject.toml library/README.md library/LICENSE ./library/
COPY library/src ./library/src
COPY scripts/isaaclab/run.py scripts/isaaclab/run.sh scripts/isaaclab/native_probe.py ./scripts/isaaclab/
RUN /isaac-sim/python.sh -m pip install \
    --constraint /opt/nexuml-isaac-constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e '.[reinforcement,tracking,video]' -e ./library \
    pytest==9.1.1 PyJWT==2.15.1 aiobotocore==2.25.1 usd-validation-nvidia==1.22.0 \
    && /isaac-sim/python.sh -m pip check \
    && /isaac-sim/python.sh -m pip freeze --all > /opt/nexuml-isaac-resolved.txt

RUN mkdir -p /workspace/output \
    && chown -R isaac-sim:isaac-sim /workspace /opt/isaaclab /var/cache/hub
USER isaac-sim
ENTRYPOINT ["/bin/bash", "/workspace/nexuml/scripts/isaaclab/run.sh"]
CMD ["-m", "nexuml", "--help"]
