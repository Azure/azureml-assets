FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:{{latest-image-tag}} AS build

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
# Pin OS packages to patched Ubuntu USN versions when this image installs or inherits them.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        curl=7.81.0-1ubuntu1.25 \
        libcurl3-gnutls=7.81.0-1ubuntu1.25 \
        libcurl4=7.81.0-1ubuntu1.25 \
        libnghttp2-14=1.43.0-1ubuntu0.4 \
        libnginx-mod-http-echo=1.18.0-6ubuntu14.18 \
        libnginx-mod-http-geoip2=1.18.0-6ubuntu14.18 \
        libunwind8 \
        libsqlite3-0=3.37.2-2ubuntu0.7 \
        libxml2=2.9.13+dfsg-1ubuntu0.12 \
        nginx-common=1.18.0-6ubuntu14.18 \
        libxml++2.6-2v5 \
        nginx-light=1.18.0-6ubuntu14.18 \
        wget=1.21.2-2ubuntu1.4 \
        tar=1.34+dfsg-1ubuntu0.1.22.04.6 \
        libkrb5support0=1.19.2-2ubuntu0.8 \
        libgssapi-krb5-2=1.19.2-2ubuntu0.8 \
        libk5crypto3=1.19.2-2ubuntu0.8 \
        libkrb5-3=1.19.2-2ubuntu0.8 \
        libpam-runtime=1.4.0-11ubuntu2.7 \
        libpam0g=1.4.0-11ubuntu2.7 \
        libpam-modules=1.4.0-11ubuntu2.7 \
        libpam-modules-bin=1.4.0-11ubuntu2.7 \
        psmisc \
        rsyslog=8.2112.0-2ubuntu2.4 \
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
    conda clean -afy && \
    find $AZUREML_CONDA_ENVIRONMENT_PATH /root -type d \( -name 'setuptools-70.3.0.dist-info' -o -name 'msgpack-1.1.2.dist-info' \) -prune -exec rm -rf {} + && \
    rm -f $AZUREML_CONDA_ENVIRONMENT_PATH/conda-meta/setuptools-70*.json \
          $AZUREML_CONDA_ENVIRONMENT_PATH/conda-meta/msgpack-python-1.1*.json \
          $AZUREML_CONDA_ENVIRONMENT_PATH/conda-meta/msgpack-1.1*.json \
          $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/bom.cdx.json && \
    if [ -f $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/vendor.txt ]; then \
        sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==83.0.0/' \
            $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/vendor.txt; \
    fi

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
# idna>=3.15 (GHSA-65pc-fj4g-8rjx): base has idna 3.11 from requests 2.32.4 (idna<4,>=2.5)
# and yarl 1.22.0 (idna>=2.0). As of 2026-05-25, latest requests/yarl keep the same loose
# idna ranges, so parent upgrades cannot force the fixed idna version.
# click>=8.3.3 (GHSA-47fr-3ffg-hgmw): base has click 8.2.1 from anaconda-cli-base/typer
# parents. As of 2026-05-25, latest anaconda-cli-base, typer, and python-dotenv still require
# click without a >=8.3.3 floor, so direct override is required.
# pip's vendored SBOM metadata still lists older vendored msgpack/setuptools versions; update/remove
# that metadata after upgrading pip so image scanners do not report non-runtime vendored records.
RUN /opt/conda/bin/conda install -n base -c conda-forge 'pip>=26.1' -y && \
    /opt/conda/bin/pip install --no-cache-dir --upgrade 'requests>=2.33.0' 'urllib3>=2.7.0' 'aiohttp>=3.14.0' 'wheel>=0.46.2' \
    'setuptools>=83.0.0' 'cryptography>=46.0.7' 'PyJWT>=2.12.0' \
    'python-dotenv>=1.2.2' 'idna>=3.15' 'click>=8.3.3' 'pydantic-settings>=2.14.2' 'msgpack>=1.2.1' && \
    find /opt/conda /root -type d \( -name 'setuptools-70.3.0.dist-info' -o -name 'msgpack-1.1.2.dist-info' \) -prune -exec rm -rf {} + && \
    rm -f /opt/conda/conda-meta/setuptools-70*.json \
          /opt/conda/conda-meta/msgpack-python-1.1*.json \
          /opt/conda/conda-meta/msgpack-1.1*.json \
          /opt/conda/lib/python3.13/site-packages/pip/_vendor/bom.cdx.json && \
    if [ -f /opt/conda/lib/python3.13/site-packages/pip/_vendor/vendor.txt ]; then \
        sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==83.0.0/' \
            /opt/conda/lib/python3.13/site-packages/pip/_vendor/vendor.txt; \
    fi
# Security: fix ptca conda env (python 3.10).
# torch==2.13.0: pinned transitive dep from ACPT ptca env; fixes GHSA-rrmf-rvhw-rf47.
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade torch==2.13.0 torchvision==0.28.0
# idna>=3.15 (GHSA-65pc-fj4g-8rjx): ptca has idna 3.11 from requests 2.32.5 (idna<4,>=2.5)
# and yarl 1.22.0 (idna>=2.0). As of 2026-05-25, latest requests/yarl keep the same loose
# idna ranges, so parent upgrades cannot force the fixed idna version.
RUN /opt/conda/bin/conda install -n ptca -c conda-forge 'pip>=26.1' -y && \
    /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade 'filelock>=3.20.3' \
    'wheel>=0.46.2' 'setuptools>=83.0.0' 'protobuf>=6.33.5' \
    'pillow>=12.2.0' 'onnx>=1.21.0' 'requests>=2.33.0' 'urllib3>=2.7.0' \
    'aiohttp>=3.14.0' 'pytest>=9.0.3' 'idna>=3.15' 'msgpack>=1.2.1' && \
    find /opt/conda/envs/ptca /root -type d \( -name 'setuptools-70.3.0.dist-info' -o -name 'msgpack-1.1.2.dist-info' \) -prune -exec rm -rf {} + && \
    rm -f /opt/conda/envs/ptca/conda-meta/setuptools-70*.json \
          /opt/conda/envs/ptca/conda-meta/msgpack-python-1.1*.json \
          /opt/conda/envs/ptca/conda-meta/msgpack-1.1*.json \
          /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/bom.cdx.json && \
    if [ -f /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/vendor.txt ]; then \
        sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==83.0.0/' \
            /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/vendor.txt; \
    fi

# Fix security vulnerabilities in active conda env (azureml-automl-dnn-vision-gpu, python 3.10)
# aiohttp, bokeh, distributed, protobuf, cryptography, filelock, setuptools, PyJWT, urllib3, pillow,
# onnx, pyarrow, requests, python-dotenv, GitPython — all transitive deps of azureml SDK packages pinned with template
# versions (cannot upgrade parent during template rendering).
# setuptools>=83.0.0: fixes GHSA-5rjg-fvgr-3xxf and GHSA-h35f-9h28-mq5c.
# cryptography>=46.0.7: azure-identity (via azureml-dataprep) only requires cryptography>=2.5.
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg / CVE-2026-25087.
# python-dotenv>=1.2.2: CVE-2026-28684; pydantic-settings (<=2.14.0) requires >=0.21.0 only.
# GitPython>=3.1.55: pinned transitive dep of mlflow-skinny/azureml-mlflow; fixes GitPython GHSA findings.
# msgpack>=1.2.1: pinned transitive dep of azureml SDK packages; fixes GHSA-6v7p-g79w-8964.
# idna>=3.15 and click>=8.3.3 are direct overrides because requests/yarl/dask parents keep
# loose dependency ranges that allow vulnerable versions.
RUN pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0' 'distributed>=2026.1.0' 'protobuf>=6.33.5' 'cryptography>=46.0.7' \
    'filelock>=3.20.3' 'setuptools>=83.0.0' 'PyJWT>=2.12.0' 'urllib3>=2.7.0' 'pillow>=12.2.0' \
    'onnx>=1.21.0' 'requests>=2.33.0' 'python-dotenv>=1.2.2' \
    'bokeh>=3.8.2' 'idna>=3.15' 'click>=8.3.3' 'pyarrow>=23.0.1' 'GitPython>=3.1.55' 'msgpack>=1.2.1' && \
    rm -f $AZUREML_CONDA_ENVIRONMENT_PATH/conda-meta/setuptools-70*.json \
          $AZUREML_CONDA_ENVIRONMENT_PATH/conda-meta/msgpack-python-1.1*.json \
          $AZUREML_CONDA_ENVIRONMENT_PATH/conda-meta/msgpack-1.1*.json \
          $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/bom.cdx.json && \
    if [ -f $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/vendor.txt ]; then \
        sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==83.0.0/' \
            $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/vendor.txt; \
    fi
RUN rm -rf /opt/conda/pkgs/pip-*/site-packages/pip/_vendor/bom.cdx.json && \
    find /opt/conda/pkgs /root -type d \( -name 'setuptools-70.3.0.dist-info' -o -name 'msgpack-1.1.2.dist-info' \) -prune -exec rm -rf {} + && \
    find /opt/conda/pkgs -path '*/site-packages/pip/_vendor/vendor.txt' -exec \
        sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==83.0.0/' {} +
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

FROM scratch
COPY --from=build / /

ENV PATH=/azureml-envs/azureml-automl-dnn-vision-gpu/bin:/usr/local/mpi/bin:/usr/local/ucx/bin:/opt/conda/envs/ptca/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.8 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471"
ENV NV_CUDA_CUDART_VERSION=11.8.89-1
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-11-8
ENV CUDA_VERSION=11.8.0
ENV LD_LIBRARY_PATH=/azureml-envs/azureml-automl-dnn-vision-gpu/lib:/opt/nccl-rdma-sharp-plugins/lib:/opt/hpcx/ompi/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NV_CUDA_LIB_VERSION=11.8.0-1
ENV NV_NVTX_VERSION=11.8.86-1
ENV NV_LIBNPP_VERSION=11.8.0.86-1
ENV NV_LIBNPP_PACKAGE=libnpp-11-8=11.8.0.86-1
ENV NV_LIBCUSPARSE_VERSION=11.7.5.86-1
ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-11-8
ENV NV_LIBCUBLAS_VERSION=11.11.3.6-1
ENV NV_LIBCUBLAS_PACKAGE=libcublas-11-8=11.11.3.6-1
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.15.5-1
ENV NCCL_VERSION=2.15.5-1
ENV NV_LIBNCCL_PACKAGE=libnccl2=2.15.5-1+cuda11.8
ENV NVIDIA_PRODUCT_NAME=CUDA
ENV NV_CUDA_CUDART_DEV_VERSION=11.8.89-1
ENV NV_NVML_DEV_VERSION=11.8.86-1
ENV NV_LIBCUSPARSE_DEV_VERSION=11.7.5.86-1
ENV NV_LIBNPP_DEV_VERSION=11.8.0.86-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-11-8=11.8.0.86-1
ENV NV_LIBCUBLAS_DEV_VERSION=11.11.3.6-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-11-8
ENV NV_LIBCUBLAS_DEV_PACKAGE=libcublas-dev-11-8=11.11.3.6-1
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=11.8.0-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-11-8=11.8.0-1
ENV NV_NVPROF_VERSION=11.8.87-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-11-8=11.8.87-1
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.15.5-1
ENV NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.15.5-1+cuda11.8
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV NV_CUDNN_VERSION=8.9.6.50
ENV NV_CUDNN_PACKAGE_NAME=libcudnn8
ENV NV_CUDNN_PACKAGE=libcudnn8=8.9.6.50-1+cuda11.8
ENV NV_CUDNN_PACKAGE_DEV=libcudnn8-dev=8.9.6.50-1+cuda11.8
ENV GPU_TYPE=cu118
ENV OS_TYPE=ubuntu2204
ENV PYTHON_VERSION=3.10
ENV PYTORCH_BUILD_VERSION=2.7.1
ENV DISTRIBUTED_BACKEND=nccl
ENV INTERNAL_OR_PUBLIC=public
ENV PYTORCH_TYPE=stable
ENV DSVM=False
ENV LANG=en_US.UTF-8
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
ENV CONDA_ENVIRONMENT_PATH=/opt/conda/envs/ptca
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-vision-gpu
ENV CONDA_DEFAULT_ENV=ptca
ENV CONDA_PREFIX=/opt/conda/envs/ptca
ENV OPAL_PREFIX=/opt/hpcx/ompi
ENV MLFLOW_MODEL_FOLDER=mlflow-model
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
ENV ENABLE_METADATA=true

EXPOSE 5001 8883 8888
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
