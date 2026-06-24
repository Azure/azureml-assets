FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260621.v1

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/lightgbm-3.3
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Update package lists and upgrade all OS packages to pull latest security patches
# (e.g. openssh USN-8222-1, curl USN-8227-1, kmod USN-8226-1, sed USN-8229-1,
# nghttp2 USN-8233-1 currently outstanding in base image openmpi5.0-ubuntu24.04:latest).
# Previously this step also did `apt-get install --reinstall -y openssl libssl-dev
# openssh-server` but those reinstalls became redundant after `apt-get upgrade` was
# added: verified 2026-05-12 against base openmpi5.0-ubuntu24.04:20260507.v1 that a
# fresh build with `apt-get upgrade` alone brings openssh to 1:9.6p1-3ubuntu13.16,
# curl/libcurl* to 8.5.0-2ubuntu10.9, libnghttp2-14 to 1.59.0-1ubuntu0.3, sed to
# 4.9-2ubuntu0.24.04.1, and libkmod2 to 31+20240202-2ubuntu7.2 - all USN fixes.
# Removed --reinstall to keep the layer minimal.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Environment setup is split into two stages (conda create + pip install) because the
# original single-step `conda env create -f conda_dependencies.yaml` combines conda's
# SAT solver with pip dependency resolution in one process. On ACR Standard build agents
# (limited memory), the conda solver consistently OOMs (exit 137) when resolving both
# conda and pip dependencies together. Splitting them avoids the memory spike.

# Step 1: Create conda env with only conda-channel packages (small, fast solve)
# pip=26.1.1: fixes CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9). Pip <26.1 ran self-update
# check after wheel install which could import newly-installed packages.
COPY conda_dependencies.yaml .
RUN conda create -p $CONDA_PREFIX -q -y python=3.10 pip=26.1.1 'pandas~=1.5.3' 'scipy~=1.10.0' 'numpy~=1.22.0' -c conda-forge -c anaconda && \
    conda clean -a -y

# Step 2: Install pip packages separately (pip resolver uses less memory than conda's SAT solver)
RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir \
    azureml-core==1.61.0.post4 \
    azureml-defaults==1.62.0 \
    azureml-mlflow==1.62.0.post3 \
    azureml-telemetry==1.62.0 \
    'scikit-learn~=1.5.1' \
    'xgboost~=1.7.6' \
    'lightgbm>=4.6.0' \
    'distributed~=2021.10.0' \
    'adlfs~=0.7.0' \
    py-spy==0.3.12 \
    'debugpy~=1.6.3' \
    'ipykernel~=6.0' \
    tensorboard \
    'psutil~=5.8.0' \
    'matplotlib~=3.5.0' \
    'tqdm~=4.66.3' \
    py-cpuinfo==5.0.0 \
    'torch-tb-profiler~=0.4.0' \
    'starlette>=1.0.1' && \
    rm conda_dependencies.yaml

# Security vulnerability fixes
# setuptools, urllib3, wheel already fixed in openmpi base (latest tag).
# pip: /opt/miniconda pip override removed: base openmpi5.0-ubuntu24.04:20260507.v1
#   now ships pip=26.1.1 in /opt/miniconda (verified 2026-05-12 via SBOM scan), so the
#   explicit `/opt/miniconda/bin/pip install --upgrade pip==26.1.1` step that fixed
#   CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9) on older base tags is now redundant. The
#   azureml-envs/lightgbm-3.3 conda env still pins pip=26.1.1 via the conda create
#   above to lock in the fix.
# cryptography: pinned transitive dep of azureml-core/azureml-mlflow; fixes GHSA-537c-gmf6-5ccf.
# aiohttp / protobuf / distributed / pyarrow: kept overrides to ensure latest patched versions;
#   parent constraints (azureml-mlflow caps mlflow-skinny<=3.9.0 -> protobuf<7,
#   dask~=2026.3.0 -> distributed<2026.3.1) would otherwise allow older patch
#   levels with outstanding CVEs, so explicit floor pins are required.
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
# starlette>=1.0.1 / idna>=3.15: transitive via fastapi / requests (base miniconda ships idna<3.15); drop when parents/base update
RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir --upgrade \
    'aiohttp>=3.14.0' 'protobuf>=6.33.5' 'distributed>=2026.1.0' cryptography==48.0.1 'idna>=3.15' 'starlette>=1.0.1' 'pyarrow>=23.0.1' && \
    /opt/miniconda/bin/pip install --no-cache-dir --upgrade 'idna>=3.15'
