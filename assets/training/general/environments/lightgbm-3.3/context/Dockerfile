FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/lightgbm-3.3
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Update package lists, upgrade all OS packages, and reinstall system OpenSSL & OpenSSH
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install --reinstall -y openssl libssl-dev openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# Security vulnerability fixes
RUN conda run -p $CONDA_PREFIX pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 'cryptography>=46.0.5' 'pillow>=12.1.1' 'protobuf>=6.33.5' 'wheel>=0.46.2' 'pip>=26.0' 'distributed>=2026.1.0' 'pyasn1>=0.6.3'
