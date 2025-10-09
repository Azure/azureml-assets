FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}

USER root

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true
RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# --- Start Vulnerability Fixes ---
RUN apt-get update && \
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
        libarchive13

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 pip=22.1.2 -y

RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH

###############################
# Pre-Build LightGBM
###############################
RUN pip install --upgrade lightgbm==4.6.0
                
###############################
# Install GPU LightGBM and XgBoost
###############################
RUN pip install --upgrade --force-reinstall xgboost==1.5.2 pandas==1.5.3

# begin conda create
# Install conda packages
RUN conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    # begin conda dependencies
    numpy~=1.23.5 \
    scikit-learn=1.5.1 \
    holidays=0.29 \
    setuptools-git \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    scipy=1.10.1 \
    joblib=1.2.0 \
    # end conda dependencies
    -c conda-forge -c anaconda

# Ensure future conda and pip calls are tied to this conda environment,
# and we don't just get lucky from the implicit PATH ordering.
SHELL ["conda", "run", "-p", "$AZUREML_CONDA_ENVIRONMENT_PATH", "/bin/bash", "-c"]
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core=={{latest-pypi-version}} \
                azureml-pipeline-core=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                'azureml-model-management-sdk==1.0.1b6.post1' \
                'azure-identity>=1.16.1' --no-cache-dir

RUN pip install \
                'inference-schema' \
                'prophet==1.1.4' \
                'pytorch-transformers==1.0.0' --no-cache-dir

RUN pip install \
                'spacy==3.7.4' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz' \
                'py-cpuinfo==5.0.0' \
                'psutil>5.0.0,<6.0.0' \
                'requests>=2.32.4' \
                'certifi>=2023.07.22' \
                'cryptography>=42.0.5' \
                'jinja2>=3.1.6' --no-cache-dir
                # end pypi dependencies

RUN HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d

# Fix vulnerabilities - GHSA-887c-mr87-cxwp: Upgrade torch to 2.8.0
RUN pip install --force-reinstall torch==2.8.0 --no-cache-dir
RUN pip install --force-reinstall torchvision==0.22.1 --no-cache-dir
RUN pip install --upgrade 'requests==2.32.4'
RUN pip install --upgrade 'urllib3==2.5.0'
RUN pip install --upgrade 'pillow==11.3.0'

# If you have other python/pip environments, update them too:
RUN /opt/miniconda/bin/pip install --upgrade 'urllib3==2.5.0' || true
RUN /opt/conda/bin/pip install --upgrade 'urllib3==2.5.0' || true
RUN $AZUREML_CONDA_ENVIRONMENT_PATH/bin/pip install --upgrade 'urllib3==2.5.0' || true
# --- End Vulnerability Fixes ---
# --- Pin NumPy <2 everywhere to avoid NumPy 2.x incompatibility ---
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install 'numpy==1.26.4' --force-reinstall && \
    pip install 'numpy==1.26.4' --force-reinstall && \
    /opt/conda/bin/pip install 'numpy==1.26.4' --force-reinstall || true
RUN $AZUREML_CONDA_ENVIRONMENT_PATH/bin/pip install --upgrade 'h2>=4.3.0'
RUN rm -rf /opt/miniconda/pkgs/


ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH


# Security vulnerability fix - GHSA-4xh5-x5gv-qwph: Upgrade pip
RUN pip install --upgrade pip
RUN /opt/conda/bin/pip install --upgrade pip || true
RUN /opt/conda/envs/ptca/bin/pip install --upgrade pip || true
RUN /opt/miniconda/bin/pip install --upgrade pip || true


