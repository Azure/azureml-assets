FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.5
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Update package lists, upgrade all OS packages, and reinstall system OpenSSL & OpenSSH
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install --reinstall -y openssl libssl-dev openssh-server libpam-runtime libc-bin dpkg-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# Security vulnerability fixes
# cryptography, pip, setuptools, urllib3, wheel already fixed in openmpi base (20260315.v1)
# cryptography: azureml-mlflow==1.62.0 (latest) caps cryptography<46.0.0, so conda env installs 44.x;
#   override after conda env create to restore 46.0.5
RUN conda run -p $CONDA_PREFIX pip install --upgrade aiohttp==3.13.3 PyNaCl==1.6.2 cryptography==46.0.5 'pyasn1>=0.6.3' 'pillow>=12.1.1' 'distributed>=2026.1.0'
