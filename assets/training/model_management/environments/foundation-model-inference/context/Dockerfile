FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Install core dependencies
RUN apt update && apt upgrade -y && \
    apt install -y software-properties-common wget git build-essential python3-dev runit && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y cuda-cudart-dev-12-4 libcublas-dev-12-4 libcusparse-dev-12-4

# Miniconda setup
ENV MINICONDA_VERSION py310_23.10.0-1
ENV PATH /opt/miniconda/bin:$PATH

RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# Create Python 3.10 environment
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/default
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 -c conda-forge --solver=classic

WORKDIR /

# Install llm-optimized-inference
RUN pip install llm-optimized-inference==0.2.31 --no-cache-dir

# Install torch
RUN pip install torch==2.7.1 --no-cache-dir

RUN pip install --use-pep517 flash-attn==2.7.4.post1

# Install flash-attn from source (more reliable than PyPI)
# RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
#    cd flash-attention && \
#    git checkout v2.7.4.post1 && \
#    pip install --no-build-isolation --no-cache-dir . && \
#    cd .. && rm -rf flash-attention

# Clean up
RUN rm -rf ~/.cache/pip

# Runit service setup
ADD runit_folder/api_server /var/runit/api_server
RUN sed -i 's/\r$//g' /var/runit/api_server/run && \
    chmod +x /var/runit/api_server/run

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001

CMD [ "runsvdir", "/var/runit" ]
