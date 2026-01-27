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
RUN conda install pip==25.3 -y
RUN conda run -n base python -m pip install --upgrade pip==25.3 urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2
# Remove all old pip metadata after upgrade
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install verl==0.6.1
RUN pip install sacrebleu==2.5.1
COPY tracking /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py

RUN pip install --no-cache-dir accelerate==1.10.0
RUN pip install --no-cache-dir sglang==0.5.4
RUN pip install --no-cache-dir sgl-kernel==0.3.16.post3

RUN pip uninstall -y mlflow 
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
RUN pip install --no-cache-dir starlette==0.49.1
RUN pip install --no-cache-dir "cryptography>=43.0.0"
# Upgrade wandb to fix golang.org/x/crypto vulnerabilities (CVE-2025-47914, CVE-2025-58181)
RUN pip install --no-cache-dir --upgrade "wandb>=0.23.0"
RUN pip install --no-cache-dir triton==3.4.0
RUN pip install xgrammar==0.1.27
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
COPY vllm_async_server dest/opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py
COPY __init__ /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/__init__.py
COPY azure_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_grader.py
COPY azure_python_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_python_grader.py
COPY utils /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/vllm/utils.py
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install vllm==0.13.0
RUN pip install openai==2.14.0
RUN pip install --force-reinstall --no-cache-dir --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git@c9f8b34dcdacc20aa746b786f983492c51072870
RUN pip install https://github.com/yeshsurya/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl
# Clean up pip caches and old package files to prevent vulnerability detection
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
