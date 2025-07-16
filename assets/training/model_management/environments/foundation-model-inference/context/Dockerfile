FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# Install necessary build tools and git
RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends \
    build-essential \
    ninja-build \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt update

# Install Python 3.10 and its development headers
RUN apt install -y python3.10 python3.10-dev python3.10-distutils

# Use python3.10 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip for Python 3.10
RUN python3.10 -m ensurepip --default-pip

# Set up Miniconda (or just use apt-installed Python)
# For simplicity and given the issue, I'm keeping your miniconda setup,
# but if you only need Python 3.10 and its packages, you could simplify
# by just using the apt-installed Python and pip.
ENV MINICONDA_VERSION py310_23.10.0-1
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/default
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge --solver=classic

ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

WORKDIR /

# Crucial: Explicitly set CUDA environment variables for flash-attn
# The -devel images usually set these, but sometimes build systems
# need them explicitly. Also, ensure TORCH_CUDA_ARCH_LIST is set
# to match your target GPU architecture.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Adjust this list based on the actual GPUs you will be running on.
# Common ones are 7.0 (V100), 7.5 (T4), 8.0 (A100), 8.6 (RTX 30 series), 8.9 (RTX 40 series), 9.0 (H100)
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Install torch before flash-attn
RUN pip install --no-cache-dir torch==2.7.1+cu124 -f https://download.pytorch.org/whl/torch_stable.html

# Install flash-attn using the recommended method from its documentation
# This ensures it builds against the correct CUDA toolkit.
# Note: flash-attn often requires a specific CUDA version alignment with PyTorch.
# torch==2.7.1+cu124 indicates CUDA 12.4, which matches your base image.

# Install other dependencies
RUN pip install llm-optimized-inference==0.2.31 --no-cache-dir

RUN pip install flash-attn==2.7.4.post1 --no-cache-dir --verbose

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

# Your runit setup
ADD runit_folder/api_server /var/runit/api_server
RUN sed -i 's/\r$//g' /var/runit/api_server/run
RUN chmod +x /var/runit/api_server/run
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001
CMD [ "runsvdir", "/var/runit" ]
