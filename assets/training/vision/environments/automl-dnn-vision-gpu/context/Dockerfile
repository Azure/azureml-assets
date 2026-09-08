FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-vision-gpu
ENV PIP_PROGRESS_BAR off
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
# Install runtime OS packages and pin vulnerable base packages to patched Ubuntu USN versions.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        coreutils=8.32-4.1ubuntu1.4 \
        curl=7.81.0-1ubuntu1.27 \
        diffutils=1:3.8-0ubuntu2.1 \
        libattr1=1:2.5.1-1ubuntu0.1 \
        libsystemd0=249.11-0ubuntu3.22 \
        libudev1=249.11-0ubuntu3.22 \
        libcurl3-gnutls=7.81.0-1ubuntu1.27 \
        libcurl4=7.81.0-1ubuntu1.27 \
        libevent-core-2.1-7=2.1.12-stable-1ubuntu0.1 \
        libgcrypt20=1.9.4-3ubuntu3.3 \
        libnghttp2-14=1.43.0-1ubuntu0.4 \
        libnginx-mod-http-echo=1.18.0-6ubuntu14.20 \
        libnginx-mod-http-geoip2=1.18.0-6ubuntu14.20 \
        libp11-kit0=0.24.0-6ubuntu0.1 \
        libssh-4=0.9.6-2ubuntu0.22.04.8 \
        libunwind8 \
        libsqlite3-0=3.37.2-2ubuntu0.7 \
        libxml2=2.9.13+dfsg-1ubuntu0.12 \
        nginx-common=1.18.0-6ubuntu14.20 \
        libxml++2.6-2v5 \
        nginx-light=1.18.0-6ubuntu14.20 \
        wget=1.21.2-2ubuntu1.5 \
        tar=1.34+dfsg-1ubuntu0.1.22.04.6 \
        libkrb5support0=1.19.2-2ubuntu0.8 \
        libgssapi-krb5-2=1.19.2-2ubuntu0.8 \
        libk5crypto3=1.19.2-2ubuntu0.8 \
        libkrb5-3=1.19.2-2ubuntu0.8 \
        libpam-runtime=1.4.0-11ubuntu2.8 \
        libpam0g=1.4.0-11ubuntu2.8 \
        libpam-modules=1.4.0-11ubuntu2.8 \
        libpam-modules-bin=1.4.0-11ubuntu2.8 \
        psmisc \
        rsyslog=8.2112.0-2ubuntu2.4 \
        runit \
        udev=249.11-0ubuntu3.22 \
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

# Security: fix torch vulnerability GHSA-rrmf-rvhw-rf47 in the active conda env.
RUN pip install --no-cache-dir --upgrade torch==2.13.0 torchvision==0.28.0
# Security: fix urllib3 (CVE-2026-37152) — transitive dep of azureml SDK
RUN pip install --no-cache-dir --upgrade 'urllib3>=2.7.0'
# Security: fix onnx (multiple CVEs) — transitive dep of azureml-automl-runtime via onnxruntime; parent pinned, cannot upgrade
RUN pip install --no-cache-dir --upgrade 'onnx>=1.21.0'

# Security: fix base conda env (python 3.13).
# pydantic-settings>=2.14.2: pinned transitive dep of azureml-defaults -> azureml-inference-server-http;
# fixes GHSA-4xgf-cpjx-pc3j.
# msgpack>=1.2.1: pinned transitive dep present in the base conda image; fixes GHSA-6v7p-g79w-8964.
# python-dotenv>=1.2.2: CVE-2026-28684 (GHSA-mf9w-mj56-hr94); transitive chain in inference layer:
#   azureml-defaults -> azureml-inference-server-http -> pydantic-settings -> python-dotenv>=0.21.0;
#   pydantic-settings (<=2.14.0) only requires >=0.21.0, no parent upgrade resolves this.
# idna>=3.15 and click>=8.3.3: parent dependency ranges permit vulnerable versions,
# so direct floors are required.
# Remove stale vendored dependency metadata after upgrading pip.
RUN /opt/conda/bin/conda install -n base -c conda-forge 'pip>=26.1' -y && \
    /opt/conda/bin/pip install --no-cache-dir --upgrade 'requests>=2.33.0' 'urllib3>=2.7.0' 'aiohttp>=3.14.0' 'wheel>=0.46.2' \
    'setuptools>=83.0.0' 'cryptography>=46.0.7' 'PyJWT>=2.12.0' \
    'python-dotenv>=1.2.2' 'idna>=3.15' 'click>=8.3.3' 'pydantic-settings>=2.14.2' 'msgpack>=1.2.1'
# Security: fix ptca conda env (python 3.10).
# torch==2.13.0: pinned transitive dep from ACPT ptca env; fixes GHSA-rrmf-rvhw-rf47.
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade torch==2.13.0 torchvision==0.28.0
# idna>=3.15: parent dependency ranges permit vulnerable versions.
RUN /opt/conda/bin/conda install -n ptca -c conda-forge 'pip>=26.1' -y && \
    /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade 'filelock>=3.20.3' \
    'wheel>=0.46.2' 'setuptools>=83.0.0' 'protobuf>=6.33.5' \
    'pillow>=12.2.0' 'onnx>=1.21.0' 'requests>=2.33.0' 'urllib3>=2.7.0' \
    'aiohttp>=3.14.0' 'pytest>=9.0.3' 'idna>=3.15' 'msgpack>=1.2.1'

# Fix security vulnerabilities in active conda env (azureml-automl-dnn-vision-gpu, python 3.10)
# aiohttp, bokeh, distributed, protobuf, cryptography, filelock, setuptools, PyJWT, urllib3, pillow,
# onnx, pyarrow, requests, python-dotenv, GitPython, sqlparse — transitive deps of azureml SDK packages pinned with template
# versions (cannot upgrade parent during template rendering).
# setuptools>=83.0.0: fixes GHSA-5rjg-fvgr-3xxf and GHSA-h35f-9h28-mq5c.
# cryptography>=46.0.7: azure-identity (via azureml-dataprep) only requires cryptography>=2.5.
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg / CVE-2026-25087.
# python-dotenv>=1.2.2: CVE-2026-28684; pydantic-settings (<=2.14.0) requires >=0.21.0 only.
# GitPython>=3.1.58: pinned transitive dep of mlflow-skinny/azureml-mlflow; fixes GitPython GHSA findings.
# sqlparse>=0.6.0: pinned transitive dep of mlflow-skinny/azureml-mlflow; fixes sqlparse GHSA findings.
# msgpack>=1.2.1: pinned transitive dep of azureml SDK packages; fixes GHSA-6v7p-g79w-8964.
# idna>=3.15 and click>=8.3.3 are direct overrides because requests/yarl/dask parents keep
# loose dependency ranges that allow vulnerable versions.
RUN pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0' 'distributed>=2026.1.0' 'protobuf>=6.33.5' 'cryptography>=46.0.7' \
    'filelock>=3.20.3' 'setuptools>=83.0.0' 'PyJWT>=2.12.0' 'urllib3>=2.7.0' 'pillow>=12.2.0' \
    'onnx>=1.21.0' 'requests>=2.33.0' 'python-dotenv>=1.2.2' \
    'bokeh>=3.8.2' 'idna>=3.15' 'click>=8.3.3' 'pyarrow>=23.0.1' 'GitPython>=3.1.58' 'sqlparse>=0.6.0' 'msgpack>=1.2.1'
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
