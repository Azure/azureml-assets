# NeMo-RL curated training environment
# Supports: GRPO, PPO, REINFORCE++, SFT
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

# NeMo-RL framework (Hydra-based CLI: nemo_rl.cli.train)
# nemo-rl is a placeholder on PyPI; install from GitHub source
RUN pip install --no-cache-dir --no-deps "git+https://github.com/NVIDIA-NeMo/RL.git@v0.2.1"

# NeMo toolkit + Megatron-Core for model parallel training
RUN pip install --no-cache-dir "nemo-toolkit[all]==2.7.0"
RUN pip install --no-cache-dir "megatron-core==0.15.3"

# Ray is required for distributed RL rollouts
RUN pip install --no-cache-dir "ray[default]==2.54.0"

# vLLM for fast rollout inference
RUN pip install --no-cache-dir "vllm==0.16.0"

# flash-attn requires torch at build time — install with --no-build-isolation
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3

# clean pip caches
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2

# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
