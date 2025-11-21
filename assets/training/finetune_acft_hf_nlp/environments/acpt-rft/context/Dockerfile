FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
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
RUN pip install --no-cache-dir pip==25.3
# Remove all old pip metadata after upgrade
RUN find /opt/conda -type d -name "pip-25.2*" -exec rm -rf {} + 2>/dev/null || true
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
COPY tracking /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py

RUN MAX_JOBS=4 pip install --no-cache-dir --upgrade flash-attn==2.8.3 --no-build-isolation
RUN pip install --no-cache-dir accelerate==1.10.0
RUN pip install --no-cache-dir sglang==0.5.4
RUN pip install --no-cache-dir triton==3.2.0
RUN pip install --no-cache-dir sgl-kernel==0.3.16.post3

RUN pip uninstall -y mlflow || true
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
RUN pip install --no-cache-dir starlette==0.49.1
RUN pip install --no-cache-dir "cryptography>=43.0.0"
# Upgrade wandb to fix golang.org/x/crypto vulnerabilities (CVE-2025-47914, CVE-2025-58181)
RUN pip install --no-cache-dir --upgrade "wandb>=0.19.0"

# Clean up pip caches and old package files to prevent vulnerability detection
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1