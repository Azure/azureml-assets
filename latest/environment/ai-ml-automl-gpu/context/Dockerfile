FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260409.v4

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
# Security: upgrade all OS packages to fix USN vulnerabilities
# Using --fix-missing to avoid failures when Ubuntu mirrors are mid-sync (systemd 404s)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade --fix-missing && \
    apt-get install -y --no-install-recommends \
        cmake \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 pip=26.0 conda-forge::tzdata -y

###############################
# Pre-Build LightGBM
###############################
RUN pip install --upgrade lightgbm==4.6.0

###############################
# Install GPU LightGBM and XgBoost
###############################
RUN pip install --upgrade --force-reinstall xgboost==1.5.2 pandas==1.5.3

# try updating pip for base env using conda
RUN conda install pip -n base -y

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
    'pip>=26.0'
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core==1.61.0.post3 \
                azureml-mlflow==1.62.0.post2 \
                azureml-pipeline-core==1.62.0 \
                azureml-telemetry==1.62.0 \
                azureml-defaults==1.62.0 \
                azureml-interpret==1.62.0 \
                azureml-responsibleai==1.62.0 \
                azureml-automl-core==1.62.0 \
                azureml-automl-runtime==1.62.0 \
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
#   Chain L1: azureml-mlflow -> mlflow-skinny<=3.5.0
#   Chain L2: azureml-train-automl-runtime -> azureml-mlflow -> mlflow-skinny
#
# bokeh>=3.8.2 — GHSA-793v-589g-574v (WebSocket origin validation bypass, CVSS 5.4)
#   conda env installs 2.4.3, pip can't auto-upgrade
#   Chain L1: azureml-train-automl-runtime -> bokeh<3.0.0
#   Chain L2: azureml-train-automl-runtime -> dask[complete] -> bokeh (via [diagnostics] extra)
RUN pip install --upgrade 'distributed>=2026.1.0' 'cryptography>=46.0.5' 'setuptools>=82.0.1' 'mlflow-skinny>=2.16.0' \
    'bokeh>=3.8.2'  # bokeh: conda env installs 2.4.3, override needed for GHSA-793v-589g-574v

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
