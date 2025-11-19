# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/o16n-base/python-assets:20240327.v1 AS inferencing-assets

# Tag: 12.3.2-cudnn9-devel-ubuntu20.04
# Env: CUDA_VERSION=12.3.2
# Env: NCCL_VERSION=2.12.7-1
# Env: NV_CUDNN_VERSION=9

# DisableDockerDetector "Preferred to use nvidia registry over MCR mirror"

FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

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
    # SSH and RDMA
    #added openssl - 1.1.1 package to mitigate vulnerabilites after cuda upgdrade, will remove this package
    libuuid1 \
    libblkid1 \
    util-linux \
    fdisk \
    libfdisk1 \
    mount \
    libmount1 \
    libsmartcols1 \
    login \
    passwd \
    libldap-common \
    openssl - 1.1.1 \
    procps \
    libprocps8 \
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    libgnutls30 \
    libtiff5 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    redis \
    iproute2 && \
    # rdma-core dependencies
    apt-get install -y \
    udev \
    libudev-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    gcc \
    ninja-build \
    pkg-config \
    valgrind \
    cython3 \
    python3-docutils \
    pandoc \
    dh-python \
    python3-dev && \
    # Others
    apt-get install -y \
    build-essential \
    bzip2 \
    libbz2-1.0 \
    systemd \
    curl \
    git \
    wget \
    cpio \
	pciutils \
	libnuma-dev \
	ibutils \
	ibverbs-utils \ 
	rdmacm-utils \
	infiniband-diags \
	perftest \
	librdmacm-dev \
	libibverbs-dev \
    libsm6 \
    libxext6 \
    libssl1.1 \
    libxrender-dev \
    libglib2.0-0 \
    dh-make \
    libc-bin \
    libx11-dev \
    libgcrypt20 \
    binutils-multiarch \
    nginx \
    e2fsprogs \
    e2fsck-static \
    fuse2fs \
    logsave \
    libss2 \
    libcom-err2 \
    gnupg \
    gnupg2 \
    gpg \
    libdpkg-perl \
    dpkg \
    libpcre3 \
    libpcre2-8-0 \
    sqlite3 \
    uidmap \
    libkrb5-26-heimdal \             
    libhcrypto4-heimdal \         
    libheimntlm0-heimdal \
    libheimbase1-heimdal \
    libasn1-8-heimdal \
    libgssapi3-heimdal \
    libhx509-5-heimdal \
    libroken18-heimdal \
    libwind0-heimdal \
    libksba8 \
    libpam-modules \
    libpam-modules-bin \
    libpam0g \
    libpam-runtime \
    tar \
    ncurses-bin \
    ncurses-base \
    libncursesw5 \
    libctf-nobfd0 \
    perl \
    fuse && \
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
RUN /var/requirements/install_system_requirements.sh && \
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
        wget -q https://github.com/openucx/ucx/releases/download/v1.9.0/ucx-1.9.0.tar.gz && \
        tar zxf ucx-1.9.0.tar.gz && \
	cd ucx-1.9.0 && \
        ./configure --prefix=/usr/local --enable-optimizations --disable-assertions --disable-params-check --enable-mt && \
        make -j $(nproc --all) && \
        make install && \
        rm -rf /tmp/ucx

# Open-MPI installation
ENV OPENMPI_VERSION 4.1.0
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
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
    curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev

#Cmake Installation
RUN apt-get update && \
    apt-get install -y cmake

# rdma-core v30.0 for Mlnx_ofed_5_1_2 as user space driver
RUN mkdir /tmp/rdma-core && \
    cd /tmp/rdma-core && \
    git clone --branch v30.0 https://github.com/linux-rdma/rdma-core && \
    cd /tmp/rdma-core/rdma-core && \
    debian/rules binary && \
    dpkg -i ../*.deb && \
    rm -rf /tmp/rdma-core

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

ENV CONDA_PREFIX=/azureml-envs/tensorflow-2.16-cuda11
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Enable debug
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

RUN conda run -p $CONDA_PREFIX

RUN pip install --no-cache-dir horovod[tensorflow]==0.28.1

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

ENV TF_USE_LEGACY_KERAS=1
