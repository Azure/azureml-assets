FROM mcr.microsoft.com/azureml/openmpi5.0-cuda13.1-ubuntu24.04:20260614.v1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_NVML_BASED_CUDA_CHECK=0

# Remove vulnerable NVIDIA Nsight Compute components
# nic_sampler (Go stdlib v1.23.4) contains 36 CVEs; this profiling plugin is not needed for inference
RUN /opt/nvidia/nsight-compute/2025.4.1/host/target-linux-x64/python/bin/python \
    -m pip install --upgrade pip==26.1.2 2>/dev/null || true && \
    rm -rf /opt/nvidia/nsight-compute/2025.4.1/host/target-linux-x64/plugins/efa_metrics

# CUDA forward-compat: lets CUDA 13.1 userspace run on hosts with older (e.g. 535-series) NVIDIA drivers.
RUN apt-get update && \
    apt-get install -y --no-install-recommends cuda-compat-13-1 && \
    rm -rf /var/lib/apt/lists/*
# Explicit list (no trailing ${LD_LIBRARY_PATH}) avoids an empty path element
# that the dynamic loader would resolve to the CWD.
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# System packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y \
        git \
        wget \
        runit

# AzureML conda environment
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/default

RUN conda create -y \
    -p ${AZUREML_CONDA_ENVIRONMENT_PATH} \
    python=3.10 \
    -c conda-forge

# Activate environment via PATH
ENV PATH=${AZUREML_CONDA_ENVIRONMENT_PATH}/bin:$PATH
ENV CONDA_DEFAULT_ENV=${AZUREML_CONDA_ENVIRONMENT_PATH}
ENV CONDA_PREFIX=${AZUREML_CONDA_ENVIRONMENT_PATH}

WORKDIR /

# When copied to assets repo, change to install from public pypi
RUN pip install llm-optimized-inference==0.2.57 --no-cache-dir

# Remove unwanted services if present
RUN rm -rf /var/runit/gunicorn \
           /var/runit/nginx \
           /var/runit/rsyslog || true

# Add API server service
ADD runit_folder/api_server /var/runit/api_server

RUN sed -i 's/\r$//g' /var/runit/api_server/run && \
    chmod +x /var/runit/api_server/run

# Validate runsvdir exists
RUN command -v runsvdir >/dev/null || \
    (echo "runsvdir missing in image" && exit 1)

ENV SVDIR=/var/runit \
    WORKER_TIMEOUT=3600

EXPOSE 5001

CMD ["runsvdir", "/var/runit"]
