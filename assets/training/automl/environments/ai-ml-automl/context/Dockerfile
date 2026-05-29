FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}


ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

# System package security upgrades.
# USN-8222-1: openssh-{client,server,sftp-server} 1:9.6p1-3ubuntu13.15 -> 1:9.6p1-3ubuntu13.16
# Parent: ubuntu 24.04 noble base image (mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04).
# `apt-get -y upgrade` alone has been observed to leave the held openssh version in
# place when an older base layer is cached, so reinstall the openssh-* packages
# explicitly to force pickup of the patched version (same pattern used in
# assets/training/aoai/proxy_components/environments/context/Dockerfile and
# assets/training/finetune_acft_hf_nlp/environments/data_import/context/Dockerfile).
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install --reinstall -y openssh-client openssh-server openssh-sftp-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# begin conda create
# Create conda environment (minimal -- packages installed via pip to avoid solver OOM)
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge && \
    conda clean -a -y

# Security: upgrade pip to fix CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9). Two pips need
# upgrading because they live in independent prefixes:
#   1. /opt/miniconda/bin/pip — ships with the parent base image
#      (mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04) at version 26.0.1.
#      VCM SCA scans this site-packages tree at
#      opt/miniconda/lib/python3.10/site-packages/pip-26.0.1.dist-info/METADATA.
#   2. $AZUREML_CONDA_ENVIRONMENT_PATH/bin/pip — created by `conda create` above.
# pip is its own parent (no upstream package can pull in a fixed pip), so explicit
# upgrades are the only available remediation.
RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade 'pip>=26.1' && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir --upgrade 'pip>=26.1'

# Install packages via pip (avoids conda solver OOM)
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir \
    'py-cpuinfo==5.0.0' \
    'joblib==1.2.0' \
    'setuptools-git' \
    'setuptools==81.0.0' \
    'psutil>5.0.0,<6.0.0' \
    'torch==2.8.0'
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-pipeline-core=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-train-automl=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                # TODO: replace the hard coded above by a referenceto azureml-train-automl[tabular]
                'mlflow-skinny==2.16.0' \
                'xgboost==1.5.2' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'inference-schema' \
                'mltable>=1.0.0' \
                'pillow==12.2.0'
                # end pypi dependencies
# end pip install

# begin pip ad-hoc
# Install pip ad-hoc dependencies for security updates
#
# distributed>=2026.1.0 (CVE-2026-23528): XSS leading to RCE via Dask dashboard
#   Chain: azureml-train-automl-runtime -> dask[complete] -> distributed
#   Parent (azureml-train-automl-runtime==1.62.0, latest) still pins dask[complete]<=2023.2.0;
#   parent fix not available, override required.
#
# bokeh>=3.8.2 (GHSA-793v-589g-574v): conda env installs 2.4.3, pip can't auto-upgrade
#   Chain: azureml-train-automl-runtime -> bokeh
#   Parent (azureml-train-automl-runtime==1.62.0, latest) still pins bokeh<3.0.0;
#   parent fix not available, override required.
#
# python-dotenv>=1.2.2 (GHSA-mf9w-mj56-hr94): set_key()/unset_key() follow symlinks
#   on cross-device .env writes, allowing arbitrary file overwrite.
#   Pulled in transitively (no direct top-level package declares it; latest
#   mlflow-skinny declares python-dotenv<2,>=0.19.0 but the pinned mlflow-skinny==2.16.0
#   does not; it is brought in by another transitive path). No parent constraint
#   blocks 1.2.2, so override directly.
RUN pip install --upgrade \
    'distributed>=2026.1.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'pillow>=12.2.0' \
    'python-dotenv>=1.2.2'
RUN pip install --no-deps --force-reinstall 'skl2onnx==1.19.1'
RUN pip install --no-deps --force-reinstall 'onnxconverter-common==1.16.0'
RUN pip install --no-deps --force-reinstall 'onnxmltools==1.14.0'
RUN pip install --no-deps --force-reinstall 'pandas==2.0.0'
# onnx: azureml-automl-runtime pins onnx<=1.17.0; override needed for
#   GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6,
#   GHSA-538c-55jv-c5g9, GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m
# pillow: upgraded from 12.1.1 for GHSA-whj4-6x5x-4v2j
# end pip ad-hoc
