FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.5
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Update package lists and reinstall system packages for security fixes
# USN-8018-2: Python regression fix (CVE-2025-15366, CVE-2025-15367, CVE-2026-0865)
# USN-8084-1: curl vulnerabilities (CVE-2026-1965, CVE-2026-3783, CVE-2026-3784)
RUN apt-get update && \
    apt-get install --reinstall -y openssl libssl-dev openssh-server libpam-runtime libc-bin dpkg-dev \
        python3.12 python3.12-minimal libpython3.12-stdlib libpython3.12-minimal \
        curl libcurl4t64 libcurl3t64-gnutls && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# Security vulnerability fixes
# cryptography: azureml-mlflow==1.62.0 (latest) caps cryptography<46.0.0, so parent upgrade cannot pull in >=46.0.5
RUN conda run -p $CONDA_PREFIX pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 cryptography==46.0.5
