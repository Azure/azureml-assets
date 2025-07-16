FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda/bin:$PATH

RUN apt update && apt upgrade -y && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y git wget runit python3.10 python3.10-dev build-essential ninja-build \
    cuda-cudart-dev-12-4 libcublas-dev-12-4 libcusparse-dev-12-4

# Manually link libcudart to avoid CUDA errors
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so || true

# Install Miniconda
ENV MINICONDA_VERSION py310_23.10.0-1
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# Create conda env
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/default
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 -c conda-forge --solver=classic
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

WORKDIR /

# Install llm-optimized-inference
RUN pip install llm-optimized-inference==0.2.31 --no-cache-dir

# Install torch 2.7.1 with CUDA 12.6 support (nightly wheel)
RUN pip install https://download.pytorch.org/whl/nightly/cu126/torch-2.7.1%2Bcu126-cp310-cp310-linux_x86_64.whl

# Install flash-attn 2.7.4.post1 from source
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.7.4.post1 && \
    pip install --no-build-isolation --no-cache-dir . && \
    cd .. && rm -rf flash-attention

# Clean cache
RUN rm -rf ~/.cache/pip

# API Server setup
ADD runit_folder/api_server /var/runit/api_server
RUN sed -i 's/\r$//g' /var/runit/api_server/run && \
    chmod +x /var/runit/api_server/run

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001
CMD [ "runsvdir", "/var/runit" ]
