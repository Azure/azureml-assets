FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.5
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Upgrade OS packages to pick up current Ubuntu security errata.
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
# pip=26.1.2: fixes CVE-2026-8643 (GHSA-wf93-45jw-7689) and CVE-2026-6357
# (GHSA-jp4c-xjxw-mgf9).
COPY conda_dependencies.yaml .
RUN conda create -p $CONDA_PREFIX -q -y python=3.10 pip=26.1.2 'pandas~=1.5.3' 'scipy~=1.10.0' 'numpy~=1.22.0' -c conda-forge -c anaconda && \
    conda clean -a -y

# Step 2: Install pip packages separately (pip resolver uses less memory than conda's SAT solver)
# starlette>=1.0.1 (GHSA-86qp-5c8j-p5mr) / idna>=3.15 (GHSA-65pc-fj4g-8rjx): transitive via fastapi / requests; drop when parent bumps
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

# Security vulnerability fixes for transitive dependencies.
# cryptography>=50.0.0: pinned transitive dep of azureml-mlflow / azureml-core; fixes GHSA-g6cj-pr64-35w5 and GHSA-537c-gmf6-5ccf.
# GitPython>=3.1.55: pinned transitive dep of azureml-mlflow; fixes GHSA-fjr4-x663-mwxc, GHSA-6p8h-3wgx-97gf, GHSA-r9mr-m37c-5fr3, GHSA-94p4-4cq8-9g67.
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir --upgrade \
    'aiohttp>=3.14.0' 'cryptography>=50.0.0' 'distributed>=2026.1.0' 'GitPython>=3.1.55' 'pyarrow>=23.0.1'

# idna>=3.15 (GHSA-65pc-fj4g-8rjx) and cryptography>=50.0.0 (GHSA-g6cj-pr64-35w5): base-image miniconda pins; drop once base image updates.
RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade 'idna>=3.15' 'cryptography>=50.0.0'
