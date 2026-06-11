# MIP-3D finetune image based on a maintained AzureML base image instead of ACPT.
#
# Base: mcr.microsoft.com/azureml/openmpi5.0-cuda12.6-ubuntu24.04
#   - Same family olympus uses
#   - On a real OS-patch cadence (vs ACPT biweekly tags shipping unpatched USNs)
#
# Pattern follows the existing azureml-assets openmpi-base curated envs (e.g.
# training/general/environments/sklearn-1.5, training/automl/.../forecasting-gpu):
#   FROM openmpi-base
#   conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10
#   conda run -p ... pip install <deps>
#
# AML training infra reads $AZUREML_CONDA_ENVIRONMENT_PATH at startup, so this
# env-path layout is effectively required for curated envs.

FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.6-ubuntu24.04:20260609.v2

USER root

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/acft-medimageparse-3d-finetune
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
# mpi needs to locate libpython from the conda env
ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

# OS deps + general housekeeping. Base is on a real cadence so apt upgrade should
# usually be a no-op (vs ACPT where we had to manually patch openssl/curl/.NET).
# NOTE: ffmpeg & its libav* / cjson / zvbi / mbedtls deps were responsible for
# 11/12 findings on the v1 openmpi scan; MIP3D is NIFTI/DICOM (no video decode),
# so we drop them entirely. libaio-dev was for deepspeed (also dropped). The
# remaining libgl/libsm/libxext/libxrender/libfontconfig set is what opencv-headless
# + matplotlib + pillow runtime need.
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

# Upgrade idna in the base miniconda Python to fix GHSA-65pc-fj4g-8rjx (idna < 3.15).
# The base image ships idna 3.13 in /opt/miniconda; pip install --upgrade pins the fix.
RUN /opt/miniconda/bin/pip install --no-cache-dir "idna>=3.15"

# Step 1: minimal conda env with python 3.10 (matches azureml-acft-image-components compat).
# Keep this step pure conda-channel — the SAT solver OOMs on ACR Standard build
# agents when conda + pip resolution run together. Sibling envs split for the
# same reason.
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH -q -y \
        python=3.10 pip=26.1 \
        -c conda-forge && \
    conda clean -a -y

# Step 2: torch from the pytorch CUDA index (split out so pip doesn't pull CPU-only wheels).
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Step 3: everything else from requirements.txt.
WORKDIR /workspace
COPY requirements.txt .
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir -r requirements.txt
