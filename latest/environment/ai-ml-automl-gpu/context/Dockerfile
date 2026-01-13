FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260111.v1

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

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 pip=25.3 conda-forge::tzdata -y

RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH

###############################
# Pre-Build LightGBM
###############################
RUN pip install --upgrade lightgbm==4.6.0

###############################
# Install GPU LightGBM and XgBoost
###############################
RUN pip install --upgrade --force-reinstall xgboost==1.5.2 pandas==1.5.3

# try updating pip for base env using conda
RUN conda install pip -n base -y || true

# begin conda create
# Create conda environment
RUN conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    # begin conda dependencies
    pip=25.3 \
    numpy~=1.23.5 \
    scikit-learn=1.5.1 \
    holidays=0.29 \
    setuptools-git \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    scipy=1.10.1 \
    'psutil>5.0.0,<6.0.0' \
    cudatoolkit=10.0.130 \
    # end conda dependencies
    -c conda-forge -c pytorch -c anaconda
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core==1.61.0.post1 \
                azureml-mlflow==1.61.0.post1 \
                azureml-pipeline-core==1.61.0 \
                azureml-telemetry==1.61.0 \
                azureml-defaults==1.61.0 \
                azureml-interpret==1.61.0 \
                azureml-responsibleai==1.61.0 \
                azureml-automl-core==1.61.0.post1 \
                azureml-automl-runtime==1.61.0 \
                azureml-train-automl-client==1.61.0.post1 \
                azureml-train-automl-runtime==1.61.0 \
                azureml-dataset-runtime==1.61.0 \
                'azureml-model-management-sdk==1.0.1b6.post1' \
                'azure-identity>=1.25.1' \
                'mlflow-skinny==2.15.1' \
                'inference-schema' \
                'py-cpuinfo==5.0.0' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'cryptography>=45.0.7' \
                'requests==2.32.5' \
                'certifi>=2025.11.12'
                # end pypi dependencies

# Upgrade requests, urllib3, pip, starlette, bokeh, PyNaCl in all relevant Python environments to meet vulnerability requirements
RUN pip install --upgrade 'starlette>=0.49.1' 'requests==2.32.5' 'urllib3==2.6.3' 'pillow==12.0.0' 'h2>=4.3.0' 'bokeh==3.8.2' 'PyNaCl==1.6.2'
RUN /opt/miniconda/bin/pip install --upgrade 'starlette>=0.49.1' 'requests==2.32.5' 'urllib3==2.6.3' 'pillow==12.0.0' 'h2>=4.3.0' 'bokeh==3.8.2' 'PyNaCl==1.6.2'

RUN rm -rf /opt/miniconda/pkgs/

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
