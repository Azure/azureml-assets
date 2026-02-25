# OpenRLHF curated training environment
# Supports: PPO, GRPO, RLOO, REINFORCE++, SFT, DPO, KTO
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install transformers==4.56.0

# OpenRLHF with vllm extra (includes ray dependency)
RUN pip install --no-cache-dir "openrlhf[vllm]>=0.5.0"

# flash-attn requires torch at build time — install with --no-build-isolation
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4

# clean conda and pip caches
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2

# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
