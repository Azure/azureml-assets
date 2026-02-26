# TorchForge (Meta) curated training environment
# Supports: SFT, GRPO via Meta's TorchForge framework (Monarch + torchtitan + vLLM)
# https://github.com/meta-pytorch/torchforge
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        openssl \
        git && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}

# TorchForge framework (Meta's post-training at scale)
# Installs: forge, torchtitan, torchmonarch, vllm, torchao, and dependencies
RUN pip install --no-cache-dir "torchforge==0.0.1"

# Clone repo for apps/ entrypoints and default configs (not included in pip package).
# torchforge's CLI is: python -m apps.<algo>.main --config apps/<algo>/<config>.yaml
RUN git clone --depth 1 https://github.com/meta-pytorch/torchforge.git /opt/torchforge
ENV PYTHONPATH="/opt/torchforge:${PYTHONPATH}"

# flash-attn requires torch at build time — install with --no-build-isolation
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3

# clean pip caches
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2

# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
