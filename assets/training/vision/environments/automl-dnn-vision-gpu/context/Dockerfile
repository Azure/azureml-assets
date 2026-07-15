FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
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
RUN pip install --no-cache-dir --upgrade torch==2.8.0 torchvision==0.23.0
# Security: fix urllib3 (CVE-2026-37152) — transitive dep of azureml SDK
RUN pip install --no-cache-dir --upgrade 'urllib3>=2.7.0'
# Security: fix onnx (multiple CVEs) — transitive dep of azureml-automl-runtime via onnxruntime; parent pinned, cannot upgrade
RUN pip install --no-cache-dir --upgrade 'onnx>=1.21.0'

# Security: fix base conda env (python 3.13).
# python-dotenv>=1.2.2: CVE-2026-28684 (GHSA-mf9w-mj56-hr94); transitive chain in inference layer:
#   azureml-defaults -> azureml-inference-server-http -> pydantic-settings -> python-dotenv>=0.21.0;
#   pydantic-settings (<=2.14.0) only requires >=0.21.0, no parent upgrade resolves this.
# idna>=3.15 (GHSA-65pc-fj4g-8rjx): base has idna 3.11 from requests 2.32.4 (idna<4,>=2.5)
# and yarl 1.22.0 (idna>=2.0). As of 2026-05-25, latest requests/yarl keep the same loose
# idna ranges, so parent upgrades cannot force the fixed idna version.
# click>=8.3.3 (GHSA-47fr-3ffg-hgmw): base has click 8.2.1 from anaconda-cli-base/typer
# parents. As of 2026-05-25, latest anaconda-cli-base, typer, and python-dotenv still require
# click without a >=8.3.3 floor, so direct override is required.
RUN /opt/conda/bin/conda install -n base -c conda-forge 'pip>=26.1' -y && \
    /opt/conda/bin/pip install --no-cache-dir --upgrade 'requests>=2.33.0' 'urllib3>=2.7.0' 'aiohttp>=3.14.0' 'wheel>=0.46.2' \
    'setuptools>=82.0.1' 'cryptography>=46.0.7' 'PyJWT>=2.12.0' \
    'python-dotenv>=1.2.2' 'idna>=3.15' 'click>=8.3.3'
# Security: fix ptca conda env (python 3.10).
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade torch==2.8.0 torchvision==0.23.0
# idna>=3.15 (GHSA-65pc-fj4g-8rjx): ptca has idna 3.11 from requests 2.32.5 (idna<4,>=2.5)
# and yarl 1.22.0 (idna>=2.0). As of 2026-05-25, latest requests/yarl keep the same loose
# idna ranges, so parent upgrades cannot force the fixed idna version.
RUN /opt/conda/bin/conda install -n ptca -c conda-forge 'pip>=26.1' -y && \
    /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade 'filelock>=3.20.3' \
    'wheel>=0.46.2' 'setuptools>=82.0.1' 'protobuf>=6.33.5' \
    'pillow>=12.2.0' 'onnx>=1.21.0' 'requests>=2.33.0' 'urllib3>=2.7.0' \
    'aiohttp>=3.14.0' 'pytest>=9.0.3' 'idna>=3.15'

# Fix security vulnerabilities in active conda env (azureml-automl-dnn-vision-gpu, python 3.10)
# aiohttp, bokeh, distributed, protobuf, cryptography, filelock, setuptools, PyJWT, urllib3, pillow, onnx, pyarrow,
# requests, python-dotenv — all transitive deps of azureml SDK packages pinned with template
# versions (cannot upgrade parent during template rendering).
# cryptography>=46.0.7: azure-identity (via azureml-dataprep) only requires cryptography>=2.5.
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg / CVE-2026-25087.
# python-dotenv>=1.2.2: CVE-2026-28684; pydantic-settings (<=2.14.0) requires >=0.21.0 only.
# idna>=3.15 and click>=8.3.3 are direct overrides because requests/yarl/dask parents keep
# loose dependency ranges that allow vulnerable versions.
RUN pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0' 'distributed>=2026.1.0' 'protobuf>=6.33.5' 'cryptography>=46.0.7' \
    'filelock>=3.20.3' 'setuptools>=82.0.1' 'PyJWT>=2.12.0' 'urllib3>=2.7.0' 'pillow>=12.2.0' \
    'onnx>=1.21.0' 'requests>=2.33.0' 'python-dotenv>=1.2.2' \
    'bokeh>=3.8.2' 'idna>=3.15' 'click>=8.3.3' 'pyarrow>=23.0.1'
# Remove stale vendored metadata that scanners pick up
RUN rm -rf /opt/conda/lib/python3.13/site-packages/setuptools/__vendor/jaraco.context-5.3.0.dist-info \
    /opt/conda/lib/python3.13/site-packages/setuptools/__vendor/wheel-0.45.1.dist-info \
    /opt/conda/envs/ptca/lib/python3.10/site-packages/setuptools/__vendor/jaraco.context-5.3.0.dist-info \
    /opt/conda/envs/ptca/lib/python3.10/site-packages/setuptools/__vendor/wheel-0.45.1.dist-info \
    /opt/conda/envs/ptca/lib/python3.10/site-packages/wheel-0.41.2.dist-info
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
