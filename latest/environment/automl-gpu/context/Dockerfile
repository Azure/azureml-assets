FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260409.v4

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true
# Upgrade critical system and python packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --only-upgrade \
        systemd \
        systemd-sysv \
        libudev1 \
        libpam-systemd \
        systemd-timesyncd \
        libsystemd0 \
        libnss-systemd \
        libpython3.10-stdlib \
        python3.10 \
        libpython3.10-minimal \
        python3.10-minimal \
        libpam0g \
        libpam-modules-bin \
        libpam-modules \
        libpam-runtime \
        libarchive13 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y
# dummy number to change when needing to force rebuild without changing the definition: 2

# Security: fix cryptography (CVE-2026-41727) — azureml-mlflow caps cryptography<46, conda env installs 44.x;
# override to restore 46.0.5 (already in openmpi base but downgraded by conda env create)
RUN pip install cryptography>=46.0.5
# fix until the base packages are updated with the new cryptography version. 
# distributed is transitively pinned at 2023.2.0 by azureml-train-automl-runtime, which depends on dask==2023.2.0 
# that in turn pulls in the matching distributed version, so it cannot be upgraded independently without updating azureml-train-automl-runtime itself.
#
# onnx>=1.21.0 — GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6,
#                GHSA-538c-55jv-c5g9, GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m
#   Parent packages cap onnx<=1.17.0; upgrading the parent is not possible because
#   both azureml-automl-runtime==1.62.0 and azureml-train-automl-runtime==1.62.0
#   (the latest releases) still enforce onnx<=1.17.0,>=1.16.1.
#   Override is required to remediate the vulnerability.
#   Chain: azureml-automl-runtime / azureml-train-automl-runtime -> onnx<=1.17.0
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --upgrade --no-cache-dir \
    'cryptography>=46.0.5' 'setuptools>=79.0.0' 'distributed>=2026.1.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0'

# Clean conda pkgs cache to remove stale vendored copies
RUN rm -rf /opt/miniconda/pkgs/

# Avoid ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AZUREML_CONDA_ENVIRONMENT_PATH/lib
