FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.5
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Update package lists and reinstall system OpenSSL & OpenSSH

# --- Security Vulnerability Fixes ---
# Fix dpkg vulnerability (USN-7768-1) for Ubuntu 24.04
RUN apt-get update && \
    apt-get install -y --only-upgrade \
    dpkg-dev=1.22.6ubuntu6.5 \
    dpkg=1.22.6ubuntu6.5 \
    libdpkg-perl=1.22.6ubuntu6.5

# Fix GNU C Library vulnerability (USN-7760-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libc-bin=2.39-0ubuntu8.6 \
    libc6-dev=2.39-0ubuntu8.6 \
    libc-dev-bin=2.39-0ubuntu8.6 \
    libc6=2.39-0ubuntu8.6

# Fix libxml2 vulnerability (USN-7743-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libxml2=2.9.14+dfsg-1.3ubuntu3.5

# Fix PAM vulnerability (USN-7761-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libpam-runtime=1.5.3-5ubuntu5.5 \
    libpam0g=1.5.3-5ubuntu5.5 \
    libpam-modules-bin=1.5.3-5ubuntu5.5 \
    libpam-modules=1.5.3-5ubuntu5.5

# Fix SQLite vulnerability (USN-7751-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libsqlite3-0=3.45.1-1ubuntu2.5

RUN apt-get update && \
    apt-get install --reinstall -y openssl libssl-dev openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y
