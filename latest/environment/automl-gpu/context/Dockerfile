FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260514.v1

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

# Security: fix cryptography (CVE-2026-41727) — conda env installs cryptography 44.x via the
# azureml-* dependency tree; override to >=46.0.5,<47.0.0. azureml-mlflow 1.62.0.post2
# (and pyopenssl 25.3.0) cap cryptography<47.0.0, so the upper bound stays within parent
# constraints. Override is still required because the conda env resolver picks an older
# 44.x build by default.
RUN pip install 'cryptography>=46.0.5,<47.0.0'

# Security: fix pip in base miniconda (CVE-2026-6357 / GHSA-jp4c-xjxw-mgf9). The conda env
# (/azureml-envs/automl) gets pip 26.1.1 via conda_dependencies.yaml, but /opt/miniconda
# still ships 26.0.1 from the base image. Upgrade the base miniconda pip directly and remove
# any leftover pip-26.0.x dist-info so the SBOM scanner does not pick up stale metadata.
RUN /opt/miniconda/bin/python -m pip install --upgrade --no-cache-dir 'pip>=26.1.1' && \
    rm -rf /opt/miniconda/lib/python3.10/site-packages/pip-26.0.*.dist-info
# distributed is transitively pinned by azureml-train-automl-runtime 1.62.0, which depends on
# dask[complete]<=2023.2.0; distributed releases track dask, so it cannot be upgraded
# independently without updating azureml-train-automl-runtime itself (1.62.0 is the latest
# release as of 2026-05-08 and still enforces the dask<=2023.2.0 cap).
# bokeh is similarly capped at <3.0.0 by azureml-train-automl-runtime 1.62.0; override required.
#
# onnx>=1.21.0 — GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6,
#                GHSA-538c-55jv-c5g9, GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m
#   Parent packages cap onnx<=1.17.0; upgrading the parent is not possible because
#   both azureml-automl-runtime==1.62.0 and azureml-train-automl-runtime==1.62.0
#   (the latest releases) still enforce onnx<=1.17.0,>=1.16.1.
#   Override is required to remediate the vulnerability.
#   Chain: azureml-automl-runtime / azureml-train-automl-runtime -> onnx<=1.17.0
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --upgrade --no-cache-dir \
    'cryptography>=46.0.5,<47.0.0' 'setuptools>=79.0.0' 'distributed>=2026.1.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0'

# Clean conda pkgs cache to remove stale vendored copies
RUN rm -rf /opt/miniconda/pkgs/

# Avoid ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AZUREML_CONDA_ENVIRONMENT_PATH/lib
