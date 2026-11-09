FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20261108.v1

# IMPORTANT (2026-10-26): Python was upgraded from 3.10 to 3.11 for Pillow 13 security fixes.
# Downstream users must update interpreter-specific paths, wheels, and compiled dependencies for Python 3.11.

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
RUN conda create -p /tmp/pmdarima-builder -y -c conda-forge \
        python=3.11 pip 'numpy>=1.23.5,<1.24' 'Cython<3' 'setuptools==59.8.0' wheel && \
    conda run -p /tmp/pmdarima-builder pip wheel --no-cache-dir --no-deps \
        --no-build-isolation --wheel-dir /opt/pmdarima-wheel 'pmdarima==1.8.5' && \
    rm -rf /tmp/pmdarima-builder

ENV PIP_FIND_LINKS=/opt/pmdarima-wheel
ENV PIP_USE_DEPRECATED=legacy-resolver

COPY conda_dependencies.yaml .
RUN /opt/miniconda/bin/python -m pip install --upgrade --no-cache-dir 'cryptography>=50.0.0' 'setuptools>=84.1.0' 'h2>=4.4.1' && \
    conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH python -m pip install --upgrade --no-cache-dir 'pip>=26.1.2' && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --upgrade --no-cache-dir \
    'cryptography>=50.0.0' 'GitPython>=3.1.55' 'setuptools>=84.1.0' 'distributed>=2026.1.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'pyarrow==23.0.1' \
    'numpy>=1.23.5,<1.24' \
    'torch==2.13.0' \
    'msgpack>=1.2.3' \
    'sqlparse>=0.6.0' && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

ENV PIP_USE_DEPRECATED=

# pmdarima 1.8.5 imports pkg_resources, which patched setuptools no longer ships.
RUN find $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.11/site-packages/pmdarima -type f -name '*.py' \
        -exec sed -i 's/from pkg_resources import parse_version/from packaging.version import parse as parse_version/' {} + && \
    ! grep -R 'from pkg_resources import parse_version' \
        $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.11/site-packages/pmdarima

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

ENV ENABLE_METADATA=true

# Upgrade vulnerable OS packages inherited from the base image.
RUN apt-get update && \
    apt-get install --only-upgrade -y --no-install-recommends \
        tar \
        libexpat1 \
        libglib2.0-0 \
        libglib2.0-data \
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
        libgcrypt20=1.9.4-3ubuntu3.3 \
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

# Normalize pip's vendored inventory after the corresponding packages are upgraded.
RUN find /opt /azureml-envs /root -path '*/pip/_vendor/vendor.txt' -type f -exec \
        sed -i -E 's/setuptools==70\.3\.0/setuptools==84.1.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g' {} + && \
    find /opt /azureml-envs /root -path '*/pip/_vendor/bom.cdx.json' -type f -delete
# Reassert newly patched transitive dependencies after all package installs.
RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade 'PyJWT>=2.15.0'

# Reassert November 2026 vulnerability-report floors in the scanned Python prefixes.
RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade 'certifi>=2026.10.19'
RUN $AZUREML_CONDA_ENVIRONMENT_PATH/bin/python -m pip install --no-cache-dir --upgrade 'opentelemetry-api>=1.45.1'

# Reassert November 2026 OS-package floors for installed packages only.
RUN set -eux; \
    apt-get update; \
    security_packages="libnginx-mod-http-echo libbrotli1"; \
    installed_packages=""; \
    for package in $security_packages; do \
        if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then \
            installed_packages="$installed_packages $package"; \
        fi; \
    done; \
    if [ -n "$installed_packages" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade $installed_packages; \
    fi; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*
