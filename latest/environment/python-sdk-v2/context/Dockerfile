FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20250619.v1

RUN apt-get update -y && apt-get install wget nscd git cpio less libglib2.0-0 libglib2.0-data util-linux binutils libnghttp2-14 libssh-4 libsqlite3-0 libpam-modules linux-libc-dev libldap-common libc-bin libc-dev-bin libc6 libc6-dev libcurl3-gnutls libgnutls30 tar openssh-server openssh-client openssl curl -y

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/python-sdk-v2
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q --solver=classic && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y
        
## Vulnerability fix
RUN pip install tqdm requests
