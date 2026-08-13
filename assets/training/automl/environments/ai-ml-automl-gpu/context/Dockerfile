FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}

USER root

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-gpu
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Pull current Ubuntu security errata; install build deps for LightGBM/XGBoost.
# Temporary base-image remediation for PAM/glibc (USN-8601-1 and USN-8611-1).
# Remove the explicit upgrades once the base image includes the patched packages.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --only-upgrade \
        libpam-runtime \
        libpam0g \
        libpam-modules \
        libpam-modules-bin \
        libc-bin \
        libc-dev-bin \
        libc6 \
        libc6-dev \
        locales && \
    apt-get install -y --no-install-recommends \
        cmake \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 'conda-forge::pip>=26.1,<27' conda-forge::tzdata -y

###############################
# Pre-Build LightGBM
###############################
RUN pip install --upgrade lightgbm==4.6.0

###############################
# Install GPU LightGBM and XgBoost
###############################
RUN pip install --upgrade --force-reinstall xgboost==1.5.2 pandas==1.5.3

# Security: upgrade pip to fix CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9). Three pip
# install paths exist on disk and each must be remediated:
#   1. /opt/miniconda/lib/python3.10/site-packages/pip-* (base miniconda from
#      the parent image at version 26.0.1)
#   2. /opt/miniconda/pkgs/pip-* (conda package cache; cleared via conda clean)
#   3. $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip-*
#      (already pinned to >=26.1 above via `conda create`)
# pip is its own parent (no upstream package can pull in a fixed pip), so explicit
# upgrades are the only available remediation. We use `conda install -n base ...`
# (rather than direct `pip install --upgrade`) so conda metadata stays consistent
# and any future `conda install -n base ...` operations don't reintroduce the old
# pip. `conda clean -a -y` then drops the now-unused 26.0.1 entry from
# /opt/miniconda/pkgs and we additionally rm any leftover pip-26.0.1* directories
# defensively because the SBOM scanner inspects that path.
RUN conda install -n base 'conda-forge::pip>=26.1,<27' -y && \
    conda clean -a -y && \
    rm -rf /opt/miniconda/pkgs/pip-26.0* /opt/miniconda/pkgs/pip-26.0.1*

# begin conda create
# Install cudatoolkit via conda (not available on pip; single-package solve is trivial)
RUN conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    cudatoolkit=10.0.130 \
    -c nvidia -c conda-forge -y

# Install scientific packages via pip (avoids conda solver OOM)
RUN pip install --no-cache-dir \
    'numpy>=1.23.5,<1.24' \
    'scikit-learn==1.5.1' \
    'holidays==0.29' \
    'setuptools-git' \
    'wheel>=0.46.2' \
    'scipy==1.10.1' \
    'psutil>5.0.0,<6.0.0' \
    'pip>=26.1,<27'
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-pipeline-core=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                'azureml-model-management-sdk==1.0.1b6.post1' \
                'inference-schema' \
                'py-cpuinfo==5.0.0' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4'

RUN pip install --no-cache-dir \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}}

RUN pip install --no-cache-dir --upgrade --no-deps 'azure-identity>=1.25.1'
                # end pypi dependencies

# ============================
# Vulnerability security fixes - transitive dependency overrides
# ============================
# Every floor below is required because the parent package still resolves a
# vulnerable version, so upgrading the parent alone does not remediate.
# distributed>=2026.1.0  CVE-2024-10096; azureml-train-automl-runtime -> dask[complete] -> distributed
# mlflow-skinny>=2.16.0  CVE-2024-37059, CVE-2025-11201; azureml-mlflow -> mlflow-skinny
# bokeh>=3.8.2           GHSA-793v-589g-574v; parent caps bokeh<3.0.0
# onnx>=1.21.0           GHSA-3r9x-f23j-gc73 and 5 related; parent caps onnx<=1.17.0
# pyarrow>=23.0.1        GHSA-rgxp-2hwp-jwgg; azureml-dataset-runtime -> pyarrow
# starlette>=1.0.1       GHSA-86qp-5c8j-p5mr; fastapi -> starlette
# idna>=3.15             GHSA-65pc-fj4g-8rjx; requests/yarl -> idna
# cryptography>=50.0.0   GHSA-537c-gmf6-5ccf, GHSA-g6cj-pr64-35w5; azureml-core/azure-identity -> cryptography
# setuptools>=83.0.0     GHSA-5rjg-fvgr-3xxf, GHSA-h35f-9h28-mq5c; azureml-automl-runtime -> pmdarima -> setuptools
# GitPython>=3.1.57      mlflow-skinny -> databricks-sdk -> gitpython keeps a loose floor
# pillow>=12.3.0         image-decoder findings via prophet -> matplotlib -> pillow
# pyasn1>=0.6.4          GHSA-hm4w-wwcw-mr6r, GHSA-8ppf-4f7h-5ppj; google-auth -> pyasn1
RUN pip install --no-cache-dir --upgrade \
    'distributed>=2026.1.0' \
    'mlflow-skinny>=2.16.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'pyarrow>=23.0.1' \
    'starlette>=1.0.1' \
    'idna>=3.15' \
    'cryptography>=50.0.0' \
    'setuptools>=83.0.0' \
    'GitPython>=3.1.57' \
    'pillow>=12.3.0' \
    'pyasn1>=0.6.4'

# The base Miniconda prefix carries its own copies of these packages, which the
# AzureML conda environment above never touches, so patch them explicitly.
RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade \
    'cryptography>=50.0.0' \
    'msgpack>=1.2.1' \
    'pydantic-settings>=2.14.2' \
    'setuptools>=83.0.0'

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
