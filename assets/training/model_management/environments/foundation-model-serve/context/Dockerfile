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

# Use the base image's miniconda Python 3.12 directly — no need to create a separate env.
RUN python3.12 -m pip install --upgrade pip==26.1

WORKDIR /

COPY . .
RUN pip install -e ./ --no-cache-dir && \
    rm -rf ~/.cache/pip

# Install our api_server service under /etc/service (empty in the base image).
# IMPORTANT: do NOT use /var/runit here — the MCR base ships gunicorn/nginx/rsyslog
# services there for its built-in scoring stack; we want only our api_server supervised.
ADD runit_folder/api_server /etc/service/api_server
RUN sed -i 's/\r$//g' /etc/service/api_server/run && \
    chmod +x /etc/service/api_server/run

ENV SVDIR=/etc/service \
    WORKER_TIMEOUT=3600
EXPOSE 5001
CMD ["runsvdir", "/etc/service"]
