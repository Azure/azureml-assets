FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# Install required system dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        software-properties-common \
        wget \
        runit \
        bash \
        git \
        build-essential \
        ninja-build \
        libopenblas-dev \
        python3-dev \
        ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV MINICONDA_VERSION py310_24.1.2-0
ENV PATH /opt/miniconda/bin:$PATH

RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    /opt/miniconda/bin/conda update --all -c conda-forge -y && \
    /opt/miniconda/bin/conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# Define conda environment
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/default
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

# Create conda environment with Python 3.10
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 -c conda-forge --solver=classic

WORKDIR /

# Install your package
RUN pip install llm-optimized-inference==0.2.31 --no-cache-dir

# Install torch with CUDA 12.1 support
RUN pip install --no-cache-dir torch==2.7.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Install flash-attn v2.7.4.post1 from source with CUDA mock (for CI without real GPU)
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.7.4.post1 && \
    pip install packaging && \
    mkdir -p /usr/local/cuda/bin && \
    echo -e '#!/bin/bash\nexit 0' > /usr/local/cuda/bin/nvcc && \
    chmod +x /usr/local/cuda/bin/nvcc && \
    pip install --use-pep517 --no-build-isolation --no-cache-dir . && \
    cd .. && rm -rf flash-attention

# Clean pip cache
RUN rm -rf ~/.cache/pip

# Setup runit
ADD runit_folder/api_server /var/runit/api_server
RUN sed -i 's/\r$//g' /var/runit/api_server/run
RUN chmod +x /var/runit/api_server/run

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600

EXPOSE 5001
CMD [ "runsvdir", "/var/runit" ]
