FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}} AS builder

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.5
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Environment setup is split into two stages (conda create + pip install) because the
# original single-step `conda env create -f conda_dependencies.yaml` combines conda's
# SAT solver with pip dependency resolution in one process. On ACR Standard build agents
# (limited memory), the conda solver consistently OOMs (exit 137) when resolving both
# conda and pip dependencies together. Splitting them avoids the memory spike.

# Step 1: Create conda env with only conda-channel packages (small, fast solve)
# pip>=26.2.0 fixes GHSA-qwm4-qh6w-59xr.
COPY conda_dependencies.yaml .
RUN conda create -p $CONDA_PREFIX -q -y python=3.10 pip=26.2.1 'pandas~=1.5.3' 'scipy~=1.10.0' 'numpy~=1.22.0' -c conda-forge -c anaconda && \
    conda clean -a -y

# Step 2: Install pip packages separately (pip resolver uses less memory than conda's SAT solver)
# starlette>=1.0.1 (GHSA-86qp-5c8j-p5mr) / idna>=3.15 (GHSA-65pc-fj4g-8rjx): transitive via fastapi / requests; drop when parent bumps
# msgpack>=1.2.1 / setuptools>=83.0.0: transitive during AzureML dependency resolution; build isolation is disabled so vulnerable build dependencies are never added in this image layer.
RUN printf '%s\n' 'msgpack>=1.2.1' 'setuptools>=83.0.0' > /tmp/pip-security-constraints.txt && \
    conda run -p $CONDA_PREFIX pip install --no-cache-dir \
    --no-build-isolation \
    --constraint /tmp/pip-security-constraints.txt \
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
    'idna>=3.15' \
    'msgpack>=1.2.1' \
    'setuptools>=83.0.0' && \
    rm conda_dependencies.yaml /tmp/pip-security-constraints.txt

# Security vulnerability fixes for transitive dependencies.
# cryptography>=50.0.0: pinned transitive dep of azureml-mlflow / azureml-core; fixes GHSA-537c-gmf6-5ccf and GHSA-g6cj-pr64-35w5.
# GitPython>=3.1.55: pinned transitive dep of azureml-mlflow; fixes GHSA-fjr4-x663-mwxc, GHSA-6p8h-3wgx-97gf, GHSA-r9mr-m37c-5fr3, GHSA-94p4-4cq8-9g67.
# msgpack>=1.2.1: pinned transitive dep of distributed; fixes GHSA-6v7p-g79w-8964.
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
# setuptools>=83.0.0: pinned transitive build/runtime dep of AzureML packages; fixes GHSA-5rjg-fvgr-3xxf and GHSA-h35f-9h28-mq5c.
# sqlparse>=0.6.0: pinned transitive dep of azureml-mlflow / mlflow; fixes GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, GHSA-pwgv-4x5q-6m9f.
RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir --upgrade \
    'aiohttp>=3.14.0' 'cryptography>=50.0.0' 'distributed>=2026.1.0' 'GitPython>=3.1.55' 'msgpack>=1.2.1' 'pyarrow>=23.0.1' 'setuptools>=83.0.0' 'sqlparse>=0.6.0' && \
    rm -f $CONDA_PREFIX/lib/python*/site-packages/pip/_vendor/bom.cdx.json $CONDA_PREFIX/lib/python*/site-packages/pip/_vendor/vendor.txt

FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.5
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Upgrade OS packages to pick up current Ubuntu security errata.
# libgcrypt20 is inherited from the base image; pin until the base image carries USN-8711-1 / CVE-2024-2236.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade libgcrypt20=1.10.3-2ubuntu0.2 && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /azureml-envs /azureml-envs
