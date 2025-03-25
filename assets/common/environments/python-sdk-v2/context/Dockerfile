FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:{{latest-image-tag}}

RUN apt-get update -y && apt-get install wget nscd git cpio less libglib2.0-0 libglib2.0-data util-linux binutils libnghttp2-14 libssh-4 libsqlite3-0 libpam-modules linux-libc-dev libldap-common libldap-2.4-2 libc-bin libc-dev-bin libc6 libc6-dev libcurl3-gnutls libgnutls30 tar openssh-server openssh-client openssl curl -y

# Install specific versions of the required Kerberos libraries
RUN apt-get install -y \
    libkrb5support0=1.17-6ubuntu4.9 \
    libgssapi-krb5-2=1.17-6ubuntu4.9 \
    libkrb5-3=1.17-6ubuntu4.9 \
    libk5crypto3=1.17-6ubuntu4.9

# Hold the packages to prevent future upgrades
RUN apt-mark hold libkrb5support0 libgssapi-krb5-2 libkrb5-3 libk5crypto3

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
