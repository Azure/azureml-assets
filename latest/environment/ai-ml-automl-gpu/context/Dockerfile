FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260614.v1

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
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
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
                azureml-core==1.61.0.post4 \
                azureml-mlflow==1.62.0.post3 \
                azureml-pipeline-core==1.62.0 \
                azureml-telemetry==1.62.0 \
                azureml-defaults==1.62.0 \
                azureml-interpret==1.62.0 \
                azureml-responsibleai==1.62.0 \
                azureml-automl-core==1.62.0.post3 \
                azureml-automl-runtime==1.62.0.post1 \
                azureml-train-automl-client==1.62.0 \
                azureml-train-automl-runtime==1.62.0 \
                azureml-dataset-runtime==1.62.0 \
                'azureml-model-management-sdk==1.0.1b6.post1' \
                'azure-identity>=1.25.1' \
                'inference-schema' \
                'py-cpuinfo==5.0.0' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4'
                # end pypi dependencies

# ============================
# Vulnerability security fixes — transitive dependency overrides
# ============================
# distributed>=2026.1.0 — CVE-2024-10096 (pickle deserialization RCE, CVSS 9.8)
#   Chain: azureml-train-automl-runtime -> dask[complete]<=2023.2.0 -> distributed==2023.2.0
#
# cryptography>=46.0.5 — CVE-2026-26007 (EC subgroup validation flaw, CVSS 8.2)
#   Chain L1: azureml-mlflow -> cryptography<47.0.0
#   Chain L1: azure-identity -> cryptography>=2.5
#   Chain L2: azureml-core -> msal/paramiko/pyopenssl/secretstorage/adal -> cryptography
#   Chain L2: azureml-mlflow -> azure-storage-blob -> cryptography
#
# setuptools>=82.0.1 — CVE-2025-47273 (PackageIndex path traversal RCE)
#   Chain: azureml-automl-runtime -> pmdarima -> setuptools
#
# mlflow-skinny>=2.16.0 — CVE-2024-37059 (unsafe deserialization, CVSS 8.8),
#                          CVE-2025-11201 (directory traversal RCE, CVSS 9.8)
#   Chain L1: azureml-mlflow==1.62.0.post2 (latest) -> mlflow-skinny<=3.9.0
#   Chain L2: azureml-train-automl-runtime -> azureml-mlflow -> mlflow-skinny
#   Override required: even the latest azureml-mlflow does not auto-pull a
#   patched mlflow-skinny.
#
# bokeh>=3.8.2 — GHSA-793v-589g-574v (WebSocket origin validation bypass, CVSS 5.4)
#   conda env installs 2.4.3, pip can't auto-upgrade
#   Chain L1: azureml-train-automl-runtime -> bokeh<3.0.0
#   Chain L2: azureml-train-automl-runtime -> dask[complete] -> bokeh (via [diagnostics] extra)
#
# onnx>=1.21.0 — GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6,
#                GHSA-538c-55jv-c5g9, GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m
#   Parent packages cap onnx<=1.17.0; upgrading the parent is not possible because
#   both azureml-automl-runtime==1.62.0 and azureml-train-automl-runtime==1.62.0
#   (the latest releases) still enforce onnx<=1.17.0,>=1.16.1.
#   Override is required to remediate the vulnerability.
#   Chain: azureml-automl-runtime / azureml-train-automl-runtime -> onnx<=1.17.0
#
# starlette>=1.0.1 — GHSA-86qp-5c8j-p5mr; transitive via fastapi (loose floor), drop when parent bumps
# idna>=3.15 — GHSA-65pc-fj4g-8rjx; transitive via requests/yarl (loose floor), drop when parent bumps
RUN pip install --upgrade 'distributed>=2026.1.0' 'cryptography>=46.0.5' 'setuptools>=82.0.1' 'mlflow-skinny>=2.16.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'starlette>=1.0.1' \
    'idna>=3.15'

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
