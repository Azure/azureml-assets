FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260408.v1


ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

# System package security upgrades
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --only-upgrade \
        libpam0g \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# begin conda create
# Create conda environment (minimal — packages installed via pip to avoid solver OOM)
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge && \
    conda clean -a -y

# Install packages via pip (avoids conda solver OOM)
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir \
    'py-cpuinfo==5.0.0' \
    'joblib==1.2.0' \
    'setuptools-git' \
    'setuptools==82.0.1' \
    'psutil>5.0.0,<6.0.0' \
    'torch==2.8.0' \
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
                azureml-interpret==1.62.0 \
                azureml-responsibleai==1.62.0 \
                azureml-automl-core==1.62.0 \
                azureml-automl-runtime==1.62.0 \
                azureml-train-automl-client==1.62.0 \
                azureml-train-automl-runtime==1.62.0 \
                azureml-train-automl==1.62.0 \
                azureml-dataset-runtime==1.62.0 \
                azureml-defaults==1.62.0 \
                # TODO: replace the hard coded above by a referenceto azureml-train-automl[tabular]
                'mlflow-skinny==2.16.0' \
                'xgboost==1.5.2' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'inference-schema' \
                'mltable>=1.0.0' \
                'pillow==12.1.1'
                # end pypi dependencies
# end pip install

# begin pip ad-hoc
# Install pip ad-hoc dependencies for security updates
#
# pip>=26.0 (CVE-2025-8869): symlink path traversal during tar extraction
#   Chain: direct conda dep (pip=26.0), upgraded for security
#
# wheel>=0.46.2 (CVE-2026-24049): path traversal & privilege escalation via malicious wheel unpack
#   Chain: setuptools[core] -> wheel (build dependency)
#
# cryptography>=46.0.5 (CVE-2026-26007): EC subgroup validation flaw in ECDH/ECDSA on SECT curves
#   Chain: mltable -> cryptography (direct dep)
#   Also:  mltable -> azureml-dataprep -> azure-identity -> cryptography
#
# protobuf>=5.29.6 (CVE-2025-4565): DoS via uncontrolled recursion in pure-Python decoder
#   Chain: mlflow-skinny -> protobuf (direct dep)
#   Also:  mlflow-skinny -> databricks-sdk -> protobuf
#
# distributed>=2026.1.0 (CVE-2026-23528): XSS leading to RCE via Dask dashboard
#   Chain: azureml-train-automl-runtime -> dask[complete] -> distributed
#
# bokeh>=3.8.2 (GHSA-793v-589g-574v): conda env installs 2.4.3, pip can't auto-upgrade
#   Chain: azureml-train-automl-runtime -> bokeh
RUN pip install --upgrade \
    'pip>=26.0' \
    'wheel>=0.46.2' \
    'cryptography>=46.0.5' \
    'protobuf>=5.29.6' \
    'distributed>=2026.1.0' \
    'bokeh>=3.8.2'  # bokeh: conda env installs 2.4.3, override needed for GHSA-793v-589g-574v
# end pip ad-hoc
