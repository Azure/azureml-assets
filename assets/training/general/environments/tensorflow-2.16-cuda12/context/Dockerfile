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
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    wget \
    zlib1g-dev \
	librdmacm-dev \
	libibverbs-dev \
    dh-make \
    # Vulnerability fixes
    libgnutls30 \
    libgssapi-krb5-2 \
    libpam-cap \
    libtasn1-6 \
    libk5crypto3 \
    libcap2  \
    gpgconf \
    # Perl vulnerability fixes
    perl-modules-5.34 \
    perl \
    perl-base \
    libperl5.34 && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* 

RUN apt-get update && \
    apt-get install -y --only-upgrade \
        libpython3.10-stdlib \
        python3.10 \
        libpython3.10-minimal \
        python3.10-minimal \
        libsqlite3-0 \
        libc6 \
        libc6-dev \
        libc-bin \
        libc-dev-bin \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        dpkg \
        dpkg-dev \
        libbinutils \
        binutils \
        binutils-common \
        binutils-multiarch \
        binutils-x86-64-linux-gnu \
        libctf0 \
        libctf-nobfd0 \
        libdpkg-perl \
        libssh-4 \
        libxml2 \
        linux-libc-dev \
        linux-headers-generic \
        locales \
        openssl \
        libssl-dev \
        libssl3 \
        gnupg2 \
        systemd \
        systemd-sysv \
        systemd-timesyncd \
        libsystemd0 \
        libnss-systemd \
        libudev1 \
        udev \
        libpam-systemd \
        libpam0g \
        libpam-modules-bin \
        libpam-modules \
        libpam-runtime \
        libpython3.10-stdlib \
        python3.10 \
        libpython3.10-minimal \
        python3.10-minimal \
        libarchive13 \
        # Vulnerabiltiy fixes
        cpp-11 \
        libtsan0 \
        libstdc++-11-dev \
        gcc-11 \
        libgcc-11-dev \
        g++-11 \
        libasan6 \
        gcc-11-base \
        libquadmath0 \
        libstdc++6 \
        libatomic1 \
        liblsan0 \
        libubsan1 \
        libcc1-0 \
        libgcc-s1 \
        libgomp1 \
        gcc-12-base \
        libitm1 \
        # USN-8233-1 (libnghttp2-14 1.43.0-1ubuntu0.3), USN-8227-1 (curl/libcurl 7.81.0-1ubuntu1.24),
        # USN-8229-1 (sed 4.8-1ubuntu2.1), USN-8226-1 (libkmod2 29-1ubuntu1.1).
        # These are root-level OS packages from individual Ubuntu source packages
        # (nghttp2/curl/sed/kmod) - no transitive parent. Fixed versions are published
        # in the jammy-security archive (uploaded 2026-04-17 .. 2026-04-30), so we install
        # them via --only-upgrade to pick the patched candidates.
        libnghttp2-14 \
        curl \
        libcurl4 \
        libcurl3-gnutls \
        sed \
        libkmod2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

               
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
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
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

#Cmake Installation
RUN apt-get update && \
    apt-get install -y cmake

# rdma-core
RUN apt-get update && \
    apt-get install -y --no-install-recommends rdma-core && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# remove unused packages
RUN apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#Install latest version of nccl-rdma-sharp-plugins
RUN cd /tmp && \
    mkdir -p /usr/local/nccl-rdma-sharp-plugins && \
    apt install -y dh-make zlib1g-dev && \
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

# Enable debug + USN-8222-1 OpenSSH upgrade
# USN-8222-1 (CVE-2026-35385/35386/35387/35388/35414) requires openssh >= 1:8.9p1-3ubuntu0.15.
# openssh-client/openssh-server/openssh-sftp-server are root-level OS packages from the
# 'openssh' Debian source package (no transitive parent); the fixed parent is published in
# the jammy-security archive (2026-04-29), so we install them directly to pick up the patched
# version (plain `apt-get install` after `apt-get update` resolves to the latest candidate).
RUN apt-get update && \
    apt-get install --reinstall -y openssl libssl-dev && \
    apt-get install -y openssh-server openssh-client openssh-sftp-server && \
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

# Security vulnerability fixes
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# setuptools vendors jaraco.context internally; >=82.0.1 bundles the patched version (GHSA-58pv-8j8x-9vj2)
# Upgrade vulnerable packages in the conda env
# cryptography>=46.0.7 fixes GHSA-m959-cc7f-wv43 and GHSA-p423-j2cm-9vmq
# requests>=2.33.0 fixes GHSA-gc5v-m9x4-r6x2
# pillow>=12.2.0 fixes GHSA-whj4-6x5x-4v2j
# pip>=26.1 fixes GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357 (bumped from >=26.0 which resolved to 26.0.1)
RUN conda run -p $CONDA_PREFIX pip install --upgrade 'pip>=26.1' 'cryptography>=46.0.7' 'setuptools>=82.0.1' 'protobuf>=5.29.6' 'requests>=2.33.0' 'pillow>=12.2.0'

# Upgrade vulnerable packages in the base miniconda env (python3.10)
# Use the absolute path to /opt/miniconda's pip because $CONDA_PREFIX/bin precedes /opt/miniconda/bin in PATH.
# pip>=26.1 fixes GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357 in the base miniconda env install path.
# idna>=3.15 fixes GHSA-65pc-fj4g-8rjx / CVE-2026-45409. `conda update --all` resolves the
# base miniconda idna to 3.13 from conda-forge, and requests (its parent) only requires
# idna<4,>=2.5, so the parent cannot pull the patched floor — direct override required.
RUN /opt/miniconda/bin/pip install --upgrade 'pip>=26.1' 'cryptography>=46.0.7' 'requests>=2.33.0' 'idna>=3.15' && \
    # Remove the conda pkgs cache — conda env create / conda install repopulate it with
    # pip-26.0.1-pyh8b19718__0 which still ships vulnerable pip 26.0.1 metadata
    # (scanned at opt/miniconda/pkgs/pip-26.0.1-pyh8b19718__0/site-packages/pip-26.0.1.dist-info).
    rm -rf /opt/miniconda/pkgs && \
    find / -type d -name __pycache__ -prune -exec rm -rf {} +

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

ENV TF_USE_LEGACY_KERAS=1
