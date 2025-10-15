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
RUN apt-get update && echo 'success updated apt-get!'
RUN apt-get update && \
    apt-get install -y --only-upgrade \
        sudo \
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

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 pip=22.1.2 conda-forge::tzdata -y

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
# Create conda environment
RUN conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    # begin conda dependencies
    numpy~=1.26.4 \
    scikit-learn=1.5.1 \
    holidays=0.29 \
    setuptools-git \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    scipy=1.10.1 \
    'psutil>5.0.0,<6.0.0' \
    pytorch=2.8.0 pytorch-cuda=12.4 \
    # end conda dependencies
    -c conda-forge -c pytorch -c anaconda -c nvidia
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
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
                'azureml-model-management-sdk==1.0.1b6.post1' \
                'azure-identity>=1.16.1' \
                'inference-schema' \
                'py-cpuinfo==5.0.0' \
                'prophet==1.1.4' \
                'cryptography>=42.0.5' \
                'requests>=2.31.0' \
                'certifi>=2023.07.22'
                # end pypi dependencies

RUN pip install --upgrade 'h2>=4.3.0'
# Install PyTorch and fix vulnerabilities
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 onnx==1.16.1
# Upgrade requests and urllib3 in all relevant Python environments to meet vulnerability requirements
RUN pip install --upgrade 'requests==2.32.4' 'urllib3==1.26.20' 'pillow==11.3.0' 'h2>=4.3.0'
RUN /opt/miniconda/bin/pip install --upgrade 'requests==2.32.4' 'urllib3==1.26.20' 'pillow==11.3.0' 'h2>=4.3.0'|| true
RUN $AZUREML_CONDA_ENVIRONMENT_PATH/bin/pip install --upgrade 'requests==2.32.4' 'urllib3==1.26.20' 'pillow==11.3.0' 'h2>=4.3.0' || true
RUN rm -rf /opt/miniconda/pkgs/

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
