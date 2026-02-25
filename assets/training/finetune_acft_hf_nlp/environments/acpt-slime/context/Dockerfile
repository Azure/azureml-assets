# Slime-RL curated training environment
# Supports: GRPO, SFT (Megatron-LM + SGLang backend)
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        openssl && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}

# Slime-RL framework (argparse CLI: slime.train)
# slime-rl is not on PyPI; install from GitHub source
RUN pip install --no-cache-dir "slime-rl>=0.2.0"

# SGLang for rollout inference (Slime uses SGLang instead of vLLM)
RUN pip install --no-cache-dir "sglang==0.5.9"
RUN pip install --no-cache-dir "sgl-kernel==0.3.21"

# Megatron-Core for tensor/pipeline parallelism
RUN pip install --no-cache-dir "megatron-core==0.15.3"

# flash-attn requires torch at build time — install with --no-build-isolation
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3

# clean pip caches
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2

# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
