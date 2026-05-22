FROM mcr.microsoft.com/azureml/openmpi5.0-cuda13.1-ubuntu24.04:{{latest-image-tag}}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_NVML_BASED_CUDA_CHECK=0

# Remove vulnerable NVIDIA Nsight Compute components
RUN rm -rf /opt/nvidia/nsight-compute

# CUDA forward-compat: lets CUDA 13.1 userspace run on hosts with older (e.g. 535-series) NVIDIA drivers.
RUN apt-get update && \
    apt-get install -y --no-install-recommends cuda-compat-13-1 && \
    rm -rf /var/lib/apt/lists/*
# Explicit list (no trailing ${LD_LIBRARY_PATH}) avoids an empty path element
# that the dynamic loader would resolve to the CWD.
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

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

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/opt/conda/envs/azureml-env

# Create conda environment
RUN conda create -y -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 -c conda-forge

# Activate environment via PATH
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

WORKDIR /


# When copied to assets repo, change to install from public pypi
RUN pip install llm-optimized-inference==0.2.56 --no-cache-dir

# clean conda and pip caches
RUN rm -rf ~/.cache/pip && \
    rm -rf /opt/miniconda/lib/python3.10/site-packages 


ADD runit_folder/api_server /etc/service/api_server
RUN sed -i 's/\r$//g' /etc/service/api_server/run && chmod +x /etc/service/api_server/run

ENV SVDIR=/etc/service
ENV WORKER_TIMEOUT=3600
EXPOSE 5001
CMD [ "runsvdir", "/etc/service" ]
