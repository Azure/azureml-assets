FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:{{latest-image-tag}}

WORKDIR /

USER root

# Install specific version to mitigate vulnerabilities in libexpat1
RUN apt-get -y update && apt-get install -y \
    libexpat1=2.2.9-1ubuntu0.7 \
    libpython3.8-stdlib=3.8.10-0ubuntu1~20.04.12 \
    libpython3.8-minimal=3.8.10-0ubuntu1~20.04.12 \
    python3.8=3.8.10-0ubuntu1~20.04.12 \
    python3.8-minimal=3.8.10-0ubuntu1~20.04.12 \
    nginx-common=1.18.0-0ubuntu1.6 \
    libnginx-mod-http-echo=1.18.0-0ubuntu1.6 \
    nginx-light=1.18.0-0ubuntu1.6 \
    libcurl3-gnutls=7.68.0-1ubuntu2.24 \
    curl=7.68.0-1ubuntu2.24 \
    libcurl4=7.68.0-1ubuntu2.24

ENV CONDA_PREFIX=/azureml-envs/mldesigner
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# Conda install and pip install could happen side by side. Remove crypytography with vulnerability from conda
RUN conda remove cryptography

# To resolve vulnerability issue regarding crytography
RUN pip install 'cryptography>=43.0.1'
