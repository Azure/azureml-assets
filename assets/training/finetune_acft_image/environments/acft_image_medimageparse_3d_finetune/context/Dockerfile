# MIP-3D finetune image based on a maintained AzureML base image.
# AML training infra reads $AZUREML_CONDA_ENVIRONMENT_PATH at startup, so this
# env-path layout is required for curated envs.

FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.6-ubuntu24.04:{{latest-image-tag}}

USER root

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/acft-medimageparse-3d-finetune
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV PIP_PROGRESS_BAR=off
# mpi needs to locate libpython from the conda env
ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

# OS deps + security updates. ffmpeg and libaio-dev are intentionally omitted:
# MIP3D uses NIFTI/DICOM inputs, and the active trainer does not use deepspeed.
# The remaining libgl/libsm/libxext/libxrender/libfontconfig set is needed by
# opencv-headless, matplotlib, and pillow at runtime.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libfontconfig1 \
        libgl1 \
        libice6 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade the vulnerable anyio package in the base Python prefix.
RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade "anyio>=4.14.2"

# Step 1: minimal conda env with python 3.10 (matches azureml-acft-image-components compat).
# Keep this step pure conda-channel — the SAT solver OOMs on ACR Standard build
# agents when conda + pip resolution run together. Sibling envs split for the
# same reason.
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH -q -y \
        python=3.10 pip=26.2.0 \
        -c conda-forge && \
    conda clean -a -y

# Step 2: PyTorch from the CUDA 12.6 index (split out so pip doesn't pull CPU-only wheels).
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0

# Step 3: everything else from requirements.txt.
WORKDIR /workspace
COPY requirements.txt .
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir -r requirements.txt

# cryptography is pinned as a transitive dep of azureml-mlflow/AzureML SDK;
# azureml-mlflow metadata caps it below the patched release, fixes GHSA-g6cj-pr64-35w5.
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir --no-deps --upgrade \
        "cryptography==50.0.0"

# Remove stale scanner metadata while retaining the upgraded installations.
RUN set -eu; \
    roots="/opt /root"; \
    [ ! -d /azureml-envs ] || roots="$roots /azureml-envs"; \
    find $roots -type f \( \
        -path '*/site-packages/pip/_vendor/bom.cdx.json' -o \
        -path '*/site-packages/pip/_vendor/vendor.txt' \
    \) -delete; \
    find $roots -type d \( \
        -name 'setuptools-70.3.0.dist-info' -o \
        -name 'setuptools-70.3.0.egg-info' -o \
        -name 'setuptools-70.3.0-py*.egg-info' -o \
        -name 'msgpack-1.1.2.dist-info' -o \
        -name 'msgpack-1.1.2.egg-info' -o \
        -name 'msgpack-1.1.2-py*.egg-info' \
    \) -prune -exec rm -rf {} +; \
    find $roots -type f \( \
        -path '*/conda-meta/setuptools-70.3.0-*.json' -o \
        -path '*/conda-meta/msgpack-python-1.1.2-*.json' -o \
        -path '*/conda-meta/msgpack-1.1.2-*.json' \
    \) -delete; \
    stale="$(find $roots \( \
        -path '*/site-packages/pip/_vendor/bom.cdx.json' -o \
        -path '*/site-packages/pip/_vendor/vendor.txt' -o \
        -name 'setuptools-70.3.0.dist-info' -o \
        -name 'setuptools-70.3.0.egg-info' -o \
        -name 'setuptools-70.3.0-py*.egg-info' -o \
        -name 'msgpack-1.1.2.dist-info' -o \
        -name 'msgpack-1.1.2.egg-info' -o \
        -name 'msgpack-1.1.2-py*.egg-info' -o \
        -path '*/conda-meta/setuptools-70.3.0-*.json' -o \
        -path '*/conda-meta/msgpack-python-1.1.2-*.json' -o \
        -path '*/conda-meta/msgpack-1.1.2-*.json' \
    \) -print -quit)"; \
    [ -z "$stale" ] || { echo "stale setuptools/msgpack scanner metadata remains: $stale" >&2; exit 1; }
