FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive
RUN apt update && apt upgrade -y && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install git -y

ENV MINICONDA_VERSION py310_23.10.0-1
ENV PATH /opt/miniconda/bin:$PATH
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget runit
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/default

# Create conda environment with py310
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge --solver=classic

ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH

ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

WORKDIR /

# When copied to assets repo, change to install from public pypi
RUN pip install llm-optimized-inference==0.2.31 --no-cache-dir

# printing
# CMD bash -c "echo 'checking CUDA info:' && nvidia-smi"

# torch installation
RUN pip install --no-cache-dir torch==2.7.1

# Ensure required build tools and dependencies for flash-attn
RUN apt-get update && \
    apt-get install -y cmake build-essential && \
    pip install --no-cache-dir packaging ninja && \
    ninja --version || (pip uninstall -y ninja && pip install ninja)

# Set CUDA environment variables (important for flash-attn build)
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN echo "Checking CUDA libs:" && ls /usr/local/cuda/lib64/libcudart.so* && \
    echo "Torch CUDA available:" && python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

# Clone and install flash-attn from specific tag
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.7.4.post1 && \
    pip install --no-build-isolation --no-cache-dir . && \
    cd .. && rm -rf flash-attention

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

ADD runit_folder/api_server /var/runit/api_server
RUN sed -i 's/\r$//g' /var/runit/api_server/run
RUN chmod +x /var/runit/api_server/run

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001
CMD [ "runsvdir", "/var/runit" ]
