FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends \
    software-properties-common \
    git wget curl gnupg2 ca-certificates lsb-release \
    build-essential cmake ninja-build \
    python3-dev libopenblas-dev && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV MINICONDA_VERSION py310_23.10.0-1
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && rm /tmp/miniconda.sh

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/default
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 -c conda-forge --solver=classic
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

# Install torch 2.7.1 with CUDA 12.1 (use correct index)
RUN pip install --no-cache-dir torch==2.7.1

# Optional: install your package
RUN pip install llm-optimized-inference==0.2.31 --no-cache-dir

# Clone and build flash-attn from source for full CUDA compatibility
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.7.4.post1 && \
    pip install packaging && \
    pip install --no-build-isolation --no-cache-dir .

# Cleanup pip cache
RUN rm -rf ~/.cache/pip

# Add runit folder (for local/server)
ADD runit_folder/api_server /var/runit/api_server
RUN sed -i 's/\r$//g' /var/runit/api_server/run && chmod +x /var/runit/api_server/run

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001
CMD [ "runsvdir", "/var/runit" ]
