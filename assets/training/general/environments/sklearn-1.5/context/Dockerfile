FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.5
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Update package lists and upgrade all OS packages to pull latest security patches
# (e.g. openssh USN-8222-1, curl USN-8227-1, kmod USN-8226-1, sed USN-8229-1,
# nghttp2 USN-8233-1 currently outstanding in base image openmpi5.0-ubuntu24.04:latest).
# Previously this step also did `apt-get install --reinstall openssl libssl-dev
# openssh-server libpam-runtime libc-bin dpkg-dev` but those reinstalls became
# redundant after `apt-get upgrade` was added: a fresh scan of the base image
# (mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:latest, digest 53c17a5a) shows
# none of those packages other than openssh have an outstanding USN, and openssh
# is already fixed by the upgrade above. Removed to keep the layer minimal.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade \
        libgnutls30t64 libgcrypt20 liblzma5 xz-utils nginx nginx-common nginx-light && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Environment setup is split into two stages (conda create + pip install) because the
# original single-step `conda env create -f conda_dependencies.yaml` combines conda's
# SAT solver with pip dependency resolution in one process. On ACR Standard build agents
# (limited memory), the conda solver consistently OOMs (exit 137) when resolving both
# conda and pip dependencies together. Splitting them avoids the memory spike.

# Step 1: Create conda env with only conda-channel packages (small, fast solve)
# pip=26.1.1: fixes CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9). Pip <26.1 ran self-update
# check after wheel install which could import newly installed packages.
COPY conda_dependencies.yaml .
RUN conda create -p $CONDA_PREFIX -q -y python=3.10 pip=26.1.1 'pandas~=1.5.3' 'scipy~=1.10.0' 'numpy~=1.22.0' -c conda-forge -c anaconda && \
    conda clean -a -y

# Step 2: Install pip packages separately (pip resolver uses less memory than conda's SAT solver)
RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir \
    scikit-learn-intelex=={{latest-pypi-version}} \
    azureml-core=={{latest-pypi-version}} \
    azureml-defaults=={{latest-pypi-version}} \
    azureml-inference-server-http=={{latest-pypi-version}} \
    azureml-mlflow=={{latest-pypi-version}} \
    azureml-telemetry=={{latest-pypi-version}} \
    'scikit-learn~=1.5.0' \
    'joblib~=1.2.0' \
    py-spy==0.3.12 \
    'debugpy~=1.6.3' \
    'ipykernel~=6.0' \
    tensorboard \
    'psutil~=5.8.0' \
    'matplotlib~=3.5.0' \
    'tqdm~=4.66.3' \
    py-cpuinfo==5.0.0 \
    'torch-tb-profiler~=0.4.0' \
    'starlette>=1.0.1' \
    'idna>=3.15' && \
    rm conda_dependencies.yaml

# Security vulnerability fixes
# pip, setuptools, urllib3, wheel already fixed in openmpi base (latest tag).
#   /opt/miniconda pip override removed: base openmpi5.0-ubuntu24.04:20260507.v1 now
#   ships pip=26.1.1 in /opt/miniconda (verified via docker inspect on 2026-05-12),
#   so the explicit `/opt/miniconda/bin/pip install --upgrade pip==26.1.1` step is
#   redundant. The azureml-envs/sklearn-1.5 conda env still pins pip=26.1.1 via the
#   conda create above to lock in the CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9) fix.
# cryptography: azureml-mlflow==1.62.0.post2 (latest) caps cryptography<47.0.0 (was
#   <46.0.0 in 1.62.0); cap no longer prevents 46.x but pip's resolver may still pick
#   an older 46.x patch via transitive constraints (azureml-core -> pyopenssl<26.0.0,
#   pyopenssl -> cryptography>=...). Pin 46.0.7 explicitly to ensure latest patch.
# aiohttp / distributed: kept overrides to ensure latest patched versions; transitive
#   parent constraints (mlflow-skinny, dask) don't require these specific patches.
RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir --upgrade \
    'aiohttp>=3.14.0' cryptography==46.0.7 'distributed>=2026.1.0'

# Upgrade idna in /opt/miniconda to fix GHSA-65pc-fj4g-8rjx (idna <3.15)
RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade 'idna>=3.15'
