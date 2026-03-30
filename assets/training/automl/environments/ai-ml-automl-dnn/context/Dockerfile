FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

# Security: upgrade all OS packages to fix USN vulnerabilities
# Using --fix-missing to avoid failures when Ubuntu mirrors are mid-sync (systemd 404s)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade --fix-missing && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# begin conda create
# Create conda environment (minimal — packages installed via pip to avoid solver OOM)
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge && \
    conda clean -a -y

# Install cudatoolkit via conda (not available on pip; single-package solve is trivial)
RUN conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    cudatoolkit=10.0.130 \
    -c nvidia -c conda-forge && \
    conda clean -a -y

# Install scientific packages via pip (avoids conda solver OOM)
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir \
    'numpy>=1.23.5,<1.24' \
    'py-cpuinfo==5.0.0' \
    'joblib==1.2.0' \
    'cloudpickle==1.6.0' \
    'scikit-learn==1.5.1' \
    'pandas>=1.5.3,<1.6' \
    'holidays==0.29' \
    'setuptools-git' \
    'setuptools==82.0.1' \
    'wheel>=0.46.2' \
    'torch==2.8.0' \
    'scipy==1.10.1' \
    'psutil>5.0.0,<6.0.0' \
    'pip>=26.0'
# end conda create

# begin pip install
# Install pip dependencies
# GitPython>=3.1.41 is required for https://github.com/advisories/GHSA-2mqj-m65w-jghx and is not available in conda
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
                azureml-dataset-runtime=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                'inference-schema' \
                'mlflow-skinny>=2.16.0' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'xgboost==1.5.2' \
                'mltable>=1.0.0' \
                'pytorch-transformers==1.0.0' \
                'GitPython>=3.1.41' \
                'spacy==3.7.4' \
                'pillow>=12.1.1' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz'
                # end pypi dependencies
# end pip install

# Fix vulnerabilities - security overrides for transitive dependencies
#
# distributed>=2026.1.0    CVE-2026-23528  XSS-to-RCE via Dask dashboard proxy
#                          Chain (L1): prophet -> distributed
#                          Chain (L1): xgboost -> distributed
#                          Chain (L2): prophet -> dask -> distributed
#
# protobuf>=5.29.6         CVE-2025-4565   DoS via recursive protobuf messages (pure-Python)
#                          Chain (L1): mlflow-skinny -> protobuf
#                          Chain (L2): azureml-automl-runtime -> onnxruntime -> protobuf
#
# cryptography>=46.0.5     CVE-2026-26007  EC subgroup validation bypass (ECDH key leak)
#                          Chain (L1): azureml-mlflow -> cryptography
#                          Chain (L1): mltable -> cryptography
#                          Chain (L2): azureml-core -> paramiko -> cryptography
#
# bokeh>=3.8.2             GHSA-793v-589g-574v  conda env installs 2.4.3, pip can't auto-upgrade
#                          Chain (L1): azureml-train-automl-runtime -> bokeh
#                          Chain (L2): prophet -> dask -> bokeh
RUN pip install --upgrade 'distributed>=2026.1.0' 'protobuf>=5.29.6' 'cryptography>=46.0.5' \
    'bokeh>=3.8.2'  # bokeh: conda env installs 2.4.3, override needed for GHSA-793v-589g-574v
RUN rm -rf /opt/miniconda/pkgs/

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
