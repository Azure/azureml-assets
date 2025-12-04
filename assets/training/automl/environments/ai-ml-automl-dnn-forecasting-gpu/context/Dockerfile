FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-forecasting-gpu
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

RUN apt-get update && \
apt-get install -y --only-upgrade \
    systemd \
    libsystemd0 \
    libnss-systemd \
    libpam-systemd \
    libudev1 \
    systemd-sysv \
    systemd-timesyncd \
    python3.10 \
    python3.10-minimal \
    libarchive13 \
    libpython3.10-minimal \
    libpython3.10-stdlib \
    libpam0g \
    libpam-modules \
    libpam-modules-bin \
    libpam-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# begin conda create
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    # begin conda dependencies
    pip=25.3 \
    numpy~=1.23.5\
    scikit-learn=1.5.1 \
    pandas~=1.5.3 \
    scipy=1.10.1 \
    'psutil>=5.2.2,<6.0.0' \
    tqdm \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    # Install pytorch separately to speed up image build
    -c conda-forge -c pytorch -c anaconda && \
    conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    pytorch=2.8.0 \
    -c pytorch -c nvidia -c conda-forge -y && \
    # end conda dependencies
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y
# end conda create

# begin pip install
# Install pip dependencies
# GitPython>=3.1.41 is required for https://github.com/advisories/GHSA-2mqj-m65w-jghx and is not available in conda
RUN pip install \
                # begin pypi dependencies
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                azureml-train-automl=={{latest-pypi-version}} \
                azureml-contrib-automl-dnn-forecasting=={{latest-pypi-version}} \
                'azure-identity>=1.25.1' \
                'inference-schema' \
                'mlflow-skinny==2.22.2' \
                'xgboost==1.5.2' \
                'cryptography>=45.0.7' \
                'requests>=2.32.5' \
                'certifi>=2025.11.12' \
                'spacy==3.7.4' \
                'GitPython>=3.1.41' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz' \
                'py-cpuinfo==5.0.0' \
                "pillow>=12.0.0"
                # end pypi dependencies
# end pip install

# Fix vulnerabilities - upgrade pip, starlette, urllib3
RUN pip install --upgrade 'starlette>=0.49.1' 'urllib3==2.5.0' 'h2>=4.3.0'
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0

RUN /bin/bash -c "source activate $AZUREML_CONDA_ENVIRONMENT_PATH && \
 export CUDACXX=/usr/local/cuda/bin/nvcc && \
 export HOROVOD_BUILD_CUDA_CC_LIST='60,61,70,75,80,86,89,90' && \
 HOROVOD_WITH_PYTORCH=1 \
 HOROVOD_CUDA_HOME=/usr/local/cuda \
 CMAKE_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib \
 pip install --no-cache-dir --no-build-isolation \
 git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d"


RUN rm -rf /opt/miniconda/pkgs/
