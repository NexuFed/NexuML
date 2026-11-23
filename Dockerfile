# syntax=docker/dockerfile:1

ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.13
ARG BASE_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

FROM $BASE_CONTAINER
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

LABEL maintainer="NexuFed AI <info@nexufed.ai>" \
    org.opencontainers.image.vendor="NexuFed AI" \
    org.opencontainers.image.licenses="Apache-2.0"

# environment settings
ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=nexuadmin
ARG UID=1000
ARG GID=1000
ENV USERNAME=${USERNAME}

# To use the default value of an ARG declared before the first FROM,
# use an ARG instruction without a value inside of a build stage:
ARG CUDA_VERSION
ARG UBUNTU_VERSION
ARG PYTHON_VERSION
ENV NEXUML_CUDA_VERSION=${CUDA_VERSION} \
    NEXUML_UBUNTU_VERSION=${UBUNTU_VERSION} \
    NEXUML_PYTHON_VERSION=${PYTHON_VERSION}

# Expose ports
EXPOSE 22 6007 8888

RUN echo "**** Installing apt packages ****"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    sudo \
    locales \
    wget \
    unzip \
    htop \
    lm-sensors \
    nvtop \
    btop  \
    nmon \
    net-tools \
    tmux \
    software-properties-common \
    libsndfile1-dev \
    sox \
    libsox-dev \
    apt-transport-https \
    gpg \
    libopenblas-openmp-dev \
    ffmpeg \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install JuiceFS
RUN add-apt-repository ppa:juicefs/ppa -y
RUN apt-get update && apt-get install -y juicefs \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "**** Setting timezone ****"

# Make the "en_US.UTF-8" locale
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG=en_US.utf8

# Setup timezone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "**** Creating user ****"
# Ubuntu 24.04 images normally reserve UID 1000 for this account.
RUN if id -u ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi
# [Optional] Set the default user. Omit if you want to keep the default as root.
RUN addgroup --gid 1000 $USERNAME
RUN adduser --disabled-password --gecos "" --uid $UID --gid $GID $USERNAME
RUN mkdir -p /home/$USERNAME
ENV HOME=/home/$USERNAME
RUN usermod -aG sudo $USERNAME
RUN echo '%sudo ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
RUN mkdir -p /env

RUN chown -R $USERNAME /env
RUN chgrp -R $USERNAME /env
RUN chown -R $USERNAME $HOME
RUN chgrp -R $USERNAME $HOME

RUN echo "**** Installing Python ${PYTHON_VERSION} ****"

# Install Python Version
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo "**** Continue as user ****"
USER $USERNAME

RUN uv venv --python ${PYTHON_VERSION} /env
ENV VIRTUAL_ENV=/env \
    UV_PROJECT_ENVIRONMENT=/env \
    UV_LINK_MODE=copy \
    PATH="/env/bin:$PATH"
WORKDIR /workspace

# Check and install locked external dependencies before copying source.
COPY --chown=${UID}:${GID} pyproject.toml ./
COPY --chown=${UID}:${GID} library/pyproject.toml library/
COPY --chown=${UID}:${GID} containers/cuda128.in containers/cuda128.txt containers/
COPY --chown=${UID}:${GID} scripts/container_lock.py scripts/
RUN --mount=type=cache,target=/home/${USERNAME}/.cache/uv,uid=${UID},gid=${GID} \
    python scripts/container_lock.py --check \
    && uv pip sync --python /env/bin/python --require-hashes --only-binary :all: \
        --torch-backend cu128 containers/cuda128.txt

COPY --chown=${UID}:${GID} . .
RUN --mount=type=cache,target=/home/${USERNAME}/.cache/uv,uid=${UID},gid=${GID} \
    uv pip install --python /env/bin/python --no-deps --editable . --editable ./library \
    && uv pip check --python /env/bin/python

CMD ["/bin/bash"]
