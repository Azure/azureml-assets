FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20250828.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

# Upgrade system-level packages to fix known vulnerabilities (systemd, python3.12, pam)
RUN apt-get update && \
    apt-get install -y --only-upgrade \
        systemd-dev \
        libudev1 \
        systemd-timesyncd \
        systemd \
        systemd-resolved \
        libsystemd0 \
        libsystemd-shared \
        libpython3.12-minimal \
        python3.12-minimal \
        libpython3.12-stdlib \
        python3.12 \
        libpam0g \
        libpam-modules-bin \
        libpam-modules \
        libpam-runtime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# begin conda create
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    # begin conda dependencies
    python=3.9 \
    pip=22.1.2 \
    numpy~=1.23.5 \
    py-cpuinfo=5.0.0 \
    joblib=1.2.0 \
    cloudpickle=1.6.0 \
    scikit-learn=1.5.1 \
    pandas~=1.5.3 \
    holidays=0.29 \
    setuptools-git \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    pytorch=2.7.1 \
    scipy=1.10.1 \
    cudatoolkit=10.0.130 \
    'psutil>5.0.0,<6.0.0' \
    # end conda dependencies
    -c conda-forge -c pytorch -c nvidia -c anaconda && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y
# end conda create

# begin pip install
# Install pip dependencies
# GitPython>=3.1.41 is required for https://github.com/advisories/GHSA-2mqj-m65w-jghx and is not available in conda
RUN pip install \
                # begin pypi dependencies
                azureml-core==1.60.0.post1 \
                azureml-mlflow==1.60.0.post1 \
                azureml-pipeline-core==1.60.0 \
                azureml-telemetry==1.60.0 \
                azureml-interpret==1.60.0 \
                azureml-responsibleai==1.60.0 \
                azureml-automl-core==1.60.0 \
                azureml-automl-runtime==1.60.0 \
                azureml-train-automl-client==1.60.0 \
                azureml-train-automl-runtime==1.60.0 \
                azureml-dataset-runtime==1.60.0 \
                azureml-defaults==1.60.0 \
                'inference-schema' \
                'prophet==1.1.4' \
                'xgboost==1.5.2' \
                'cryptography>=42.0.5' \
                'mltable>=1.0.0' \
                'pytorch-transformers==1.0.0' \
                'GitPython>=3.1.41' \
                'spacy==3.7.4' \
                'pillow>=11.3.0' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz'
                # end pypi dependencies
# end pip install

# Fix vulnerabilities
RUN pip install --upgrade urllib3
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

