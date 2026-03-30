FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260329.v1

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

# Environment setup is split into two stages (conda create + pip install) because the
# original single-step `conda env create -f conda_dependencies.yaml` combines conda's
# SAT solver with pip dependency resolution in one process. On ACR Standard build agents
# (limited memory), the conda solver consistently OOMs (exit 137) when resolving both
# conda and pip dependencies together. Splitting them avoids the memory spike.

# Step 1: Create conda env with only conda-channel packages (small, fast solve)
COPY conda_dependencies.yaml .
RUN conda create -p $CONDA_PREFIX -q -y python=3.10 pip=26.0 'pandas~=1.5.3' 'scipy~=1.10.0' 'numpy~=1.22.0' -c conda-forge -c anaconda && \
    conda clean -a -y

# Step 2: Install pip packages separately (pip resolver uses less memory than conda's SAT solver)
RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir \
    scikit-learn-intelex==2025.11.0 \
    azureml-core==1.61.0.post3 \
    azureml-defaults==1.62.0 \
    azureml-inference-server-http==1.5.1 \
    azureml-mlflow==1.62.0.post2 \
    azureml-telemetry==1.62.0 \
    'scikit-learn~=1.5.0' \
    'joblib~=1.2.0' \
    py-spy==0.3.12 \
    'debugpy~=1.6.3' \
    'ipykernel~=6.0' \
    tensorboard \
    'psutil~=5.8.0' \
    'matplotlib~=3.5.0' \
    'tqdm~=4.66.3' \
    py-cpuinfo==5.0.0 \
    'torch-tb-profiler~=0.4.0' \
    'starlette>=0.49.1' && \
    rm conda_dependencies.yaml

# Security vulnerability fixes
# cryptography, pip, setuptools, urllib3, wheel already fixed in openmpi base (20260315.v1)
# cryptography: azureml-mlflow==1.62.0 (latest) caps cryptography<46.0.0, so conda env installs 44.x;
#   override after conda env create to restore 46.0.5
RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir --upgrade \
    aiohttp==3.13.3 cryptography==46.0.5 'distributed>=2026.1.0'
