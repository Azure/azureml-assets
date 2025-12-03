FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202509.2
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        openssl && \
    apt-get install -y --only-upgrade \
        linux-headers-generic \
        linux-libc-dev \
        libsystemd0 \
        libudev1 \
        udev \
        systemd \
        base-files \
        bash \
        coreutils \
        tar \
        wget \
        curl \
        patch \
        ncurses-base \
        ncurses-bin \
        login \
        passwd \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
        libc6 \
        libc-bin \
        util-linux \
        libpcre2-8-0 \
        grep \
        gzip || true && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*.deb
RUN conda install pip==25.3 -y
RUN conda run -n base python -m pip install --upgrade pip==25.3
# Remove all old pip metadata after upgrade
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-acft-common-components==0.0.83
RUN pip install azureml-evaluate-mlflow==0.0.83
RUN pip install verl==0.6.0
COPY tracking /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py
COPY vllm_async_server dest/opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py


RUN pip install --no-cache-dir accelerate==1.10.0
RUN pip install --no-cache-dir sglang==0.5.4

RUN pip install --no-cache-dir sgl-kernel==0.3.16.post3

RUN pip uninstall -y mlflow || true
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
RUN pip install --no-cache-dir starlette==0.49.1
RUN pip install --no-cache-dir "cryptography>=43.0.0"
# Upgrade wandb to fix golang.org/x/crypto vulnerabilities (CVE-2025-47914, CVE-2025-58181)
RUN pip install --no-cache-dir --upgrade "wandb>=0.23.0"
RUN pip install vllm==0.11.0
RUN pip install --no-cache-dir triton==3.4.0
RUN pip install xgrammar==0.1.27
RUN MAX_JOBS=$(nproc) pip install --no-cache-dir --upgrade flash-attn==2.8.3 --no-build-isolation
# Clean up pip caches and old package files to prevent vulnerability detection
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
