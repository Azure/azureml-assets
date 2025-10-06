FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}


ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

# System package security upgrades

# --- Security Vulnerability Fixes ---
# Fix dpkg vulnerability (USN-7768-1) for Ubuntu 24.04
RUN apt-get update && \
    apt-get install -y --only-upgrade \
    dpkg-dev=1.22.6ubuntu6.5 \
    dpkg=1.22.6ubuntu6.5 \
    libdpkg-perl=1.22.6ubuntu6.5

# Fix GNU C Library vulnerability (USN-7760-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libc-bin=2.39-0ubuntu8.6 \
    libc6-dev=2.39-0ubuntu8.6 \
    libc-dev-bin=2.39-0ubuntu8.6 \
    libc6=2.39-0ubuntu8.6

# Fix libxml2 vulnerability (USN-7743-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libxml2=2.9.14+dfsg-1.3ubuntu3.5

# Fix PAM vulnerability (USN-7761-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libpam-runtime=1.5.3-5ubuntu5.5 \
    libpam0g=1.5.3-5ubuntu5.5 \
    libpam-modules-bin=1.5.3-5ubuntu5.5 \
    libpam-modules=1.5.3-5ubuntu5.5

# Fix SQLite vulnerability (USN-7751-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libsqlite3-0=3.45.1-1ubuntu2.5

RUN apt-get update && \
    apt-get install -y --only-upgrade \
        libpam0g \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# begin conda create
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.9 \
    # begin conda dependencies
    pip \
    py-cpuinfo=5.0.0 \
    joblib=1.2.0 \
    setuptools-git \
    setuptools=78.1.1 \
    'psutil>5.0.0,<6.0.0' \
    pytorch=2.7.1 \
    # end conda dependencies
    -c conda-forge -c pytorch -c anaconda -c nvidia && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y
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
                'mlflow-skinny==2.15.1' \
                'xgboost==1.5.2' \
                'prophet==1.1.4' \
                'inference-schema' \
                'mltable>=1.0.0' \
                'pillow==11.3.0'
                # end pypi dependencies
# end pip install

# begin pip ad-hoc
# Install pip ad-hoc dependencies for security updates
RUN pip install --upgrade 'pyarrow==14.0.2'
RUN pip install torch==2.7.1 torchvision==0.22.1
RUN pip install urllib3==2.5.0
# end pip ad-hoc

