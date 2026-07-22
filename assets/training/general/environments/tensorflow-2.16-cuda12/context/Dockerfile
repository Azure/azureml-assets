# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 AS inferencing-assets

# Tag: 12.8.1-cudnn-devel-ubuntu22.04
# Env: CUDA_VERSION=12.8.1
# Env: NCCL_VERSION=2.12.7-1
# Env: NV_CUDNN_VERSION=9

# DisableDockerDetector "Preferred to use nvidia registry over MCR mirror"

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

USER root:root

ARG IMAGE_NAME=None
ARG BUILD_NUMBER=None

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

# Install Common Dependencies
# apt-get upgrade -y clears the base image's overdue Ubuntu OS-package CVEs (kerberos,
# glibc, libtasn1, sqlite, perl, binutils, gnupg, dpkg, libcap2, pam, etc.). Run before the
# UCX/Open-MPI/Horovod source builds so they compile against the patched toolchain.
RUN apt-get -o Acquire::Retries=5 update && \
    apt-get -o Acquire::Retries=5 upgrade -y && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    curl \
    git \
    wget \
    zlib1g-dev \
	librdmacm-dev \
	libibverbs-dev \
    dh-make && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* 

# Update to latest redis
RUN apt-get update && apt-get install -y lsb-release && \
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list && \
    apt-get update && apt-get install -y redis

# Inference
# Copy logging utilities, nginx and rsyslog configuration files, IOT server binary, etc.
COPY --from=inferencing-assets /artifacts /var/
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
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
    ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=300
EXPOSE 5001 8883 8888
# Stores image version information and log it while running inferencing server for better Debuggability
RUN if [ "$BUILD_NUMBER" != "None" ] && [ "$IMAGE_NAME" != "None" ]; then echo "${IMAGE_NAME}, Materializaton Build:${BUILD_NUMBER}" > /IMAGE_INFORMATION ; fi

# Conda Environment
ENV MINICONDA_VERSION py310_23.10.0-1
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda > /tmp/miniconda-install.log 2>&1 || (cat /tmp/miniconda-install.log && false) && \
    conda update --all -c conda-forge -y -q && \
    conda clean -ay -q && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf   

# Open-MPI-UCX installation
RUN mkdir /tmp/ucx && \
    cd /tmp/ucx && \
        wget -q https://github.com/openucx/ucx/releases/download/v1.17.0/ucx-1.17.0.tar.gz && \
        tar zxf ucx-1.17.0.tar.gz && \
	cd ucx-1.17.0 && \
        ./configure --prefix=/usr/local --enable-optimizations --disable-assertions --disable-params-check --enable-mt && \
        make -j $(nproc --all) && \
        make install && \
        rm -rf /tmp/ucx

# Open-MPI installation
ENV OPENMPI_VERSION 4.1.0
RUN mkdir /tmp/openmpi && cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --with-ucx=/usr/local/ --enable-mca-no-build=btl-uct --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi
    	
# Msodbcsql17 installation
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev

# cmake + rdma-core + autoremove in one layer to avoid redundant apt-get update calls
RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get install -y --no-install-recommends rdma-core && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#Install latest version of nccl-rdma-sharp-plugins
RUN cd /tmp && \
    mkdir -p /usr/local/nccl-rdma-sharp-plugins && \
    git clone -b v2.1.0 https://github.com/Mellanox/nccl-rdma-sharp-plugins.git && \
    cd nccl-rdma-sharp-plugins && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/nccl-rdma-sharp-plugins --with-cuda=/usr/local/cuda --without-ucx && \
    make && \
    make install

# set env var to find nccl rdma plugins inside this container
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl-rdma-sharp-plugins/lib

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/tensorflow-2.16-cuda12
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# USN-8222-1 (CVE-2026-35385/35386/35387/35388/35414): openssh >= 1:8.9p1-3ubuntu0.15.
# Installed directly from jammy-security to pick up patched openssh-{server,client,sftp-server}.
RUN apt-get update && \
    apt-get install -y openssh-server openssh-client openssh-sftp-server > /tmp/openssh-install.log 2>&1 || (cat /tmp/openssh-install.log && false) && \
    apt-get clean && \
    rm -rf /tmp/openssh-install.log /var/lib/apt/lists/*

# Upgrade vulnerable OS packages installed by the Ubuntu/CUDA/inferencing base layers:
# OpenSSL (USN-8414-1), systemd (USN-8402-1), xz-utils (USN-8362-1),
# libgcrypt (USN-8319-1), libarchive (USN-8292-1), and GnuTLS (USN-8284-1).
RUN apt-get update && \
    apt-get install --only-upgrade -y --no-install-recommends \
        libarchive13 \
        libgcrypt20 \
        libgnutls30 \
        liblzma5 \
        libnss-systemd \
        libpam-systemd \
        libssl-dev \
        libssl3 \
        libsystemd0 \
        libudev1 \
        openssl \
        systemd \
        systemd-sysv \
        systemd-timesyncd \
        udev \
        xz-utils > /tmp/os-security-upgrade.log 2>&1 || (cat /tmp/os-security-upgrade.log && false) && \
    apt-get clean && \
    rm -rf /tmp/os-security-upgrade.log /var/lib/apt/lists/*

# USN-8458-1: nginx packages >= 1.18.0-6ubuntu14.16. nginx-light is installed in the
# inferencing layer above (after the top apt-get upgrade), so the nginx packages are
# re-upgraded here.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nginx-common \
        nginx-light \
        libnginx-mod-http-geoip2 \
        libnginx-mod-http-echo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade current Ubuntu USN fixes for packages installed by the CUDA and inferencing layers.
RUN apt-get update && \
    apt-get install --only-upgrade -y --no-install-recommends \
        curl \
        gzip \
        libcurl3-gnutls \
        libcurl4 \
        libnghttp2-14 \
        libpython3.10-minimal \
        libpython3.10-stdlib \
        libsqlite3-0 \
        libxml2 \
        python3.10 \
        python3.10-minimal \
        tar && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y


RUN HOROVOD_WITH_TENSORFLOW=1  HOROVOD_CUDA_HOME=/usr/local/cuda pip  install --no-cache-dir --no-build-isolation horovod["tensorflow"]==0.28.1

# Reinstall OpenSSL inside Conda
RUN conda run -p $CONDA_PREFIX conda install -c conda-forge openssl

# Security vulnerability fixes in the conda env (azureml-mlflow pins cryptography<46; overridden for the CVE):
# cryptography>=46.0.7 fixes GHSA-m959-cc7f-wv43 and GHSA-p423-j2cm-9vmq
# setuptools>=82.0.1 fixes GHSA-58pv-8j8x-9vj2
# requests>=2.33.0 fixes GHSA-gc5v-m9x4-r6x2
# pillow>=12.2.0 fixes GHSA-whj4-6x5x-4v2j
# pip>=26.1 fixes GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357 (bumped from >=26.0 which resolved to 26.0.1)
# starlette>=1.0.1 (GHSA-86qp-5c8j-p5mr) / idna>=3.15 (GHSA-65pc-fj4g-8rjx): transitive via fastapi / requests; drop when parent bumps
# PyJWT>=2.13.0 fixes GHSA-993g-76c3-p5m4 and GHSA-jq35-7prp-9v3f
# pyarrow>=23.0.1 fixes GHSA-rgxp-2hwp-jwgg
RUN conda run -p $CONDA_PREFIX pip install --upgrade 'pip>=26.1' 'cryptography>=46.0.7' 'setuptools>=82.0.1' 'protobuf>=5.29.6' 'requests>=2.33.0' 'pillow>=12.2.0' 'starlette>=1.0.1' 'idna>=3.15' 'PyJWT>=2.13.0' 'pyarrow>=23.0.1'

# Upgrade vulnerable packages in the base miniconda env (python3.10)
# Use the absolute path to /opt/miniconda's pip because $CONDA_PREFIX/bin precedes /opt/miniconda/bin in PATH.
# pip>=26.1 fixes GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357 in the base miniconda env install path.
# py-rattler is a transitive dep of conda-rattler-solver in the Miniconda base env; remove it after
# build-time conda use because the current wheel embeds vulnerable pyo3 (GHSA-36hh-v3qg-5jq4/GHSA-chgr-c6px-7xpp).
RUN /opt/miniconda/bin/pip install --upgrade \
        'pip>=26.1' \
        'cryptography>=46.0.7' \
        'requests>=2.33.0' \
        'idna>=3.15' && \
    conda config --system --set solver classic && \
    /opt/miniconda/bin/pip uninstall -y py-rattler conda-rattler-solver && \
    rm -rf /opt/miniconda/lib/python3.10/site-packages/rattler* \
           /opt/miniconda/lib/python3.10/site-packages/conda_rattler_solver* \
           /opt/miniconda/lib/python3.10/site-packages/py__rattler* \
           /opt/miniconda/lib/python3.10/site-packages/py_rattler* \
           /opt/miniconda/conda-meta/py-rattler-*.json \
           /opt/miniconda/conda-meta/conda-rattler-solver-*.json && \
    # Remove the conda pkgs cache — conda env create / conda install repopulate it with
    # pip-26.0.1-pyh8b19718__0 which still ships vulnerable pip 26.0.1 metadata
    # (scanned at opt/miniconda/pkgs/pip-26.0.1-pyh8b19718__0/site-packages/pip-26.0.1.dist-info).
    rm -rf /opt/miniconda/pkgs && \
    find / -type d -name __pycache__ -prune -exec rm -rf {} +

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

ENV TF_USE_LEGACY_KERAS=1
