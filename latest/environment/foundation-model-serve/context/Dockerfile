FROM mcr.microsoft.com/azureml/openmpi5.0-cuda13.1-ubuntu24.04:20260514.v1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_NVML_BASED_CUDA_CHECK=0

# Remove vulnerable NVIDIA Nsight Compute components if present in the base.
RUN rm -rf /opt/nvidia/nsight-compute || true

# CUDA forward-compat: cuda-compat-13-1 ships pre-installed in the MCR base at
# /usr/local/cuda-13.1/compat (reachable via /usr/local/cuda -> /etc/alternatives/cuda).
# Prepending it to LD_LIBRARY_PATH lets the compat libcuda be preferred over the
# host-injected driver, so hosts with older NVIDIA drivers (e.g. 535-series) can
# still run CUDA 13.x userspace.
# An explicit absolute value (no ${LD_LIBRARY_PATH} suffix) avoids the empty-element
# loader risk if LD_LIBRARY_PATH is unset at runtime.
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

# git is needed by some pip editable installs / VCS deps.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Create the AzureML-conventional conda environment so app deps stay isolated from the
# base image's /opt/miniconda and downstream tooling can find /azureml-envs/default.
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/default
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.12 \
    -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    find / -type d -name __pycache__ -prune -exec rm -rf {} +

ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH \
    CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH \
    CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

RUN python3.12 -m pip install --upgrade pip==26.1

WORKDIR /

COPY . .
# Use `python3.12 -m pip` (not bare `pip`) so the install always targets the
# /azureml-envs/default interpreter even if PATH ordering changes later.
RUN python3.12 -m pip install -e ./ --no-cache-dir && \
    rm -rf ~/.cache/pip

# Install our api_server service under /etc/service (empty in the base image).
# IMPORTANT: do NOT use /var/runit here — the MCR base ships gunicorn/nginx/rsyslog
# services there for its built-in scoring stack; we want only our api_server supervised.
ADD runit_folder/api_server /etc/service/api_server
RUN sed -i 's/\r$//g' /etc/service/api_server/run && \
    chmod +x /etc/service/api_server/run

ENV SVDIR=/etc/service \
    WORKER_TIMEOUT=3600

# install runit as the CMD below depends on `runsvdir`).
RUN command -v runsvdir >/dev/null || { echo 'runsvdir missing in base image'; exit 1; }

EXPOSE 5001
CMD ["runsvdir", "/etc/service"]
