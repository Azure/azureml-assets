FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:{{latest-image-tag}}


ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        libc-bin \
        dpkg-dev \
        libssl-dev \
        dpkg \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        binutils \
        binutils-common \
        binutils-x86-64-linux-gnu \
        libbinutils \
        libctf0 \
        libctf-nobfd0 \
        libc6 \
        libc6-dev \
        libc-dev-bin \
        libssh-4 \
        libxml2 \
        linux-libc-dev \
        linux-headers-generic \
        locales \
        openssl \
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*&& \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

ENV ENABLE_METADATA=true

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda clean -afy

# Install packages with torch packages separately to reduce layer size
RUN pip install --no-cache-dir \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-automl-dnn-vision=={{latest-pypi-version}}

# Security: fix torch remote code execution (CVE-2025-32434)
RUN pip install --upgrade torch==2.8.0 torchvision==0.23.0
# Security: fix urllib3 (CVE-2026-37152) — transitive dep of azureml SDK
RUN pip install --upgrade 'urllib3>=2.6.3'
# Security: fix onnx (multiple CVEs) — transitive dep of azureml-automl-runtime via onnxruntime; parent pinned, cannot upgrade
RUN pip install --upgrade 'onnx>=1.21.0'

# Security: fix base conda env (python 3.13) — packages present in base image at vulnerable versions.
# Base image (probed) contains: aiohttp 3.12.14, cryptography 44.0.1, pip 25.3, PyJWT 2.10.1,
# python-dotenv 1.1.0, requests 2.32.4, setuptools 80.9.0, urllib3 2.5.0, wheel 0.45.1.
# python-dotenv>=1.2.2: CVE-2026-28684 (GHSA-mf9w-mj56-hr94); transitive chain in inference layer:
#   azureml-defaults -> azureml-inference-server-http -> pydantic-settings -> python-dotenv>=0.21.0;
#   pydantic-settings (<=2.14.0) only requires >=0.21.0, no parent upgrade resolves this.
RUN /opt/conda/bin/pip install --no-cache-dir --upgrade 'requests>=2.33.0' 'urllib3>=2.6.3' 'aiohttp>=3.13.4' 'wheel>=0.46.2' \
    'setuptools>=82.0.1' 'cryptography>=46.0.7' 'PyJWT>=2.12.0' 'pip>=26.1' \
    'python-dotenv>=1.2.2'
# Security: fix ptca conda env (python 3.10) — packages present in base image at vulnerable versions.
# Base image (probed) contains: aiohttp 3.13.3, filelock 3.20.0, onnx 1.20.1, pillow 12.1.0,
# pip 25.3, protobuf 6.33.3, pytest 7.4.3, requests 2.32.5, setuptools 80.9.0, urllib3 2.6.3,
# wheel 0.41.2, torch 2.7.1+cu118, torchvision 0.22.1+cu118.
# urllib3 already at required 2.6.3 in base — no override needed.
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade torch==2.8.0 torchvision==0.23.0 
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade 'filelock>=3.20.3' \
    'wheel>=0.46.2' 'setuptools>=82.0.1' 'protobuf>=6.33.5' 'pip>=26.1' \
    'pillow>=12.2.0' 'onnx>=1.21.0' 'requests>=2.33.0' \
    'aiohttp>=3.13.4' 'pytest>=9.0.3'

# Fix security vulnerabilities in active conda env (azureml-automl-dnn-vision-gpu, python 3.10)
# aiohttp, bokeh, distributed, protobuf, cryptography, filelock, wheel, setuptools, PyJWT,
# urllib3, pillow, onnx, requests, python-dotenv — all transitive deps of azureml SDK packages
# pinned with template versions (cannot upgrade parent during template rendering).
# python-dotenv>=1.2.2: CVE-2026-28684; pydantic-settings (<=2.14.0) requires >=0.21.0 only.
RUN pip install --upgrade 'aiohttp>=3.13.4' 'distributed>=2026.1.0' 'protobuf>=6.33.5' 'pip>=26.1' 'cryptography>=46.0.7' \
    'filelock>=3.20.3' 'wheel>=0.46.2' 'setuptools>=82.0.1' 'PyJWT>=2.12.0' 'urllib3>=2.6.3' 'pillow>=12.2.0' \
    'onnx>=1.21.0' 'requests>=2.33.0' 'python-dotenv>=1.2.2' \
    'bokeh>=3.8.2'
# Remove stale vendored metadata that scanners pick up
RUN rm -rf /opt/conda/lib/python3.13/site-packages/setuptools/__vendor/jaraco.context-5.3.0.dist-info \
    /opt/conda/lib/python3.13/site-packages/setuptools/__vendor/wheel-0.45.1.dist-info \
    /opt/conda/envs/ptca/lib/python3.10/site-packages/setuptools/__vendor/jaraco.context-5.3.0.dist-info \
    /opt/conda/envs/ptca/lib/python3.10/site-packages/setuptools/__vendor/wheel-0.45.1.dist-info \
    /opt/conda/envs/ptca/lib/python3.10/site-packages/wheel-0.41.2.dist-info


ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
# dummy number to change when needing to force rebuild without changing the definition: 1
