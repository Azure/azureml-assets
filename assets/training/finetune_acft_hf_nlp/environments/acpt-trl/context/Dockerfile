# TRL curated training environment
# Supports: SFT, DPO, KTO, GRPO, RLOO, ORPO, CPO, SimPO, RewardModel
# Unified environment replacing acft-hf-nlp-gpu and acft-group-relative-policy-optimization
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

COPY requirements.txt .
# The below 2 files are required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
COPY finetune_run.py /azureml/finetune/run.py

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install nltk==3.9.1
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN MAX_JOBS=$(nproc) pip install --no-cache-dir --upgrade flash-attn==2.8.3 --no-build-isolation

# vulnerabilities, cannot be added to requirements.txt as it causes pip dependency resolver to break
RUN pip install --upgrade --no-cache-dir 'fastmcp>=2.14.0'

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2

# Clean up pip caches
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
