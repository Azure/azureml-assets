FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/lightgbm-3.3
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Update package lists, upgrade all OS packages, and reinstall system OpenSSL & OpenSSH
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install --reinstall -y openssl libssl-dev openssh-server && \
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
    azureml-core=={{latest-pypi-version}} \
    azureml-defaults=={{latest-pypi-version}} \
    azureml-mlflow=={{latest-pypi-version}} \
    azureml-telemetry=={{latest-pypi-version}} \
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
    'starlette>=0.49.1' && \
    rm conda_dependencies.yaml

# Security vulnerability fixes
# setuptools, urllib3, wheel already fixed in openmpi base (latest tag).
# pip: openmpi base (20260409.v4) still ships pip=26.0.1 in /opt/miniconda which is
#   vulnerable to CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9). Upgrade /opt/miniconda's pip to
#   26.1.1 explicitly here. (The azureml-envs/lightgbm-3.3 conda env is created with
#   pip=26.1.1 via the conda create command above.)
# cryptography: azureml-core==1.61.0.post3 pulls pyopenssl<26.0.0 which caps cryptography
#   transitively, causing the conda env to resolve to 45.0.7 (vulnerable to CVE-2026-26007,
#   CVE-2026-34073, CVE-2026-39892 / GHSA-r6ph-v2qm-q3c2, GHSA-m959-cc7f-wv43,
#   GHSA-p423-j2cm-9vmq). azureml-mlflow==1.62.0.post2 caps cryptography<47.0.0, so pin
#   46.0.7 explicitly (latest within the cap). Parent package (azureml-core) cannot be
#   upgraded past 1.61.0.post3 as that is the latest released version.
# aiohttp / protobuf / distributed: kept overrides to ensure latest patched versions;
#   parent constraints (mlflow-skinny -> protobuf<6, dask -> distributed) cap older
#   versions so direct upgrade is required.
RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade pip==26.1.1 && \
    conda run -p $CONDA_PREFIX pip install --no-cache-dir --upgrade \
    'aiohttp>=3.13.5' 'protobuf>=6.33.5' 'distributed>=2026.1.0' cryptography==46.0.7
