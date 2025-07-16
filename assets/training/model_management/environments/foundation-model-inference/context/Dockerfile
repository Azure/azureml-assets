# Start with a base image that includes CUDA 12.6.
# You will need to verify the exact tag for CUDA 12.6 on NVIDIA's Docker Hub (https://hub.docker.com/r/nvidia/cuda/tags)
# once it is officially released. I'm using '12.6.0-devel-ubuntu22.04' as a likely candidate.
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

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

# Install llm-optimized-inference after PyTorch, so it can leverage the CUDA-enabled torch.
RUN pip install llm-optimized-inference==0.2.31 --no-cache-dir

# torch installation with CUDA 12.6 support
# This command is crucial for installing the CUDA 12.6 compatible version.
# Ensure torchvision and torchaudio versions are compatible with torch==2.7.1.
RUN pip install --no-cache-dir \
    torch==2.7.1+cu126 \
    torchvision==0.18.1+cu126 \
    torchaudio==2.7.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# Verify the PyTorch CUDA installation
# This step helps confirm that CUDA is correctly detected by PyTorch inside the container.
RUN python3 -c "import torch; \
    print(f'PyTorch version: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    if torch.cuda.is_available(): print(f'CUDA version: {torch.version.cuda}'); \
    else: print('CUDA is NOT available. Check base image and installation.'); \
    assert torch.cuda.is_available(), 'CUDA is not detected by PyTorch!'"

# flash-attn install
# Building from source will typically link against the PyTorch installation in the environment,
# so installing PyTorch with CUDA 12.6 first is important.
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.7.4.post1 && \
    pip install packaging && \
    # --no-build-isolation ensures it uses the currently active environment's PyTorch
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
