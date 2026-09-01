FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260901.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# distributed/bokeh/onnx/msgpack: parent packages cap these below patched versions.
# GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6,
# GHSA-538c-55jv-c5g9, GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m (onnx)
# cryptography, setuptools, and h2 are inherited from the base image; cryptography is also a transitive dep of azureml-core.
# Fixes GHSA-g6cj-pr64-35w5, GHSA-h35f-9h28-mq5c, and GHSA-6hr6-w5qg-qmwg.
# pyarrow is a pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
# torch is a pinned transitive dep of azureml-contrib-automl-dnn-forecasting; fixes GHSA-rrmf-rvhw-rf47.
# msgpack is a pinned transitive dep of distributed; fixes GHSA-6v7p-g79w-8964.
# sqlparse is a pinned transitive dep of AzureML AutoML packages; fixes GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, GHSA-pwgv-4x5q-6m9f.
COPY conda_dependencies.yaml .
RUN /opt/miniconda/bin/python -m pip install --upgrade --no-cache-dir 'cryptography>=50.0.0' 'setuptools>=83.0.0' 'h2>=4.4.1' && \
    conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH python -m pip install --upgrade --no-cache-dir 'pip>=26.1.2' && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --upgrade --no-cache-dir \
    'cryptography>=50.0.0' 'GitPython>=3.1.55' 'setuptools>=83.0.0' 'distributed>=2026.1.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'pyarrow==23.0.1' \
    'torch==2.13.0' \
    'msgpack>=1.2.1' \
    'sqlparse>=0.6.0' && \
    rm -rf $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip-26.1.1.dist-info && \
    rm -f $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/bom.cdx.json \
        $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/vendor.txt && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

ENV ENABLE_METADATA=true

RUN apt-get update && \
    apt-get install --only-upgrade -y --no-install-recommends \
        tar \
        libxml2 \
        curl \
        libcurl3-gnutls \
        libcurl4 \
        libsqlite3-0 \
        nginx-light \
        libnginx-mod-http-echo \
        nginx-common \
        libnginx-mod-http-geoip2 \
        wget \
        gzip \
        libkrb5support0 \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        openssh-client \
        openssh-server \
        openssh-sftp-server \
        libpam-runtime \
        libpam0g \
        libpam-modules \
        libpam-modules-bin \
        libssl3 \
        openssl \
        libperl5.34 \
        perl \
        perl-base \
        perl-modules-5.34 \
        rsyslog \
        libc-dev-bin \
        libc6-dev \
        libc6 \
        libc-bin && \
    rm -rf /var/lib/apt/lists/*

RUN rm -rf /opt/miniconda/pkgs/

# Avoid ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AZUREML_CONDA_ENVIRONMENT_PATH/lib