FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# Upgrade pip in base conda env (CVE-2026-1703 requires pip>=26.0)
RUN conda install -n base -c conda-forge 'pip>=26.0' -y

# Security upgrades in BASE conda env (Python 3.13). Parent packages in base image
# (anaconda-auth, anaconda-cli-base, conda, conda-content-trust, requests, typer, yarl,
#  SecretStorage) do not yet pin to patched versions; direct overrides required.
RUN conda run -n base python -m pip install --upgrade \
    'wheel>=0.46.2' \
    'cryptography>=46.0.7' \
    'setuptools>=82.0.1' \
    'PyJWT>=2.12.0' \
    'aiohttp>=3.13.4' \
    'urllib3>=2.7.0' \
    'requests>=2.33.0' \
    'python-dotenv>=1.2.2' \
    'idna>=3.15' \
    'click>=8.3.3'

# Upgrade pip in ptca conda env (CVE-2026-1703 requires pip>=26.0)
RUN conda install -c conda-forge 'pip>=26.0' -y && \
    conda clean -a -y

# Install pip dependencies (runs in default PTCA conda env)
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir


# Inference requirements
# libcurl4, liblttng-ust1, openssh-client are already installed by the base image -- omitted.
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# Deepspeed launcher requires passwordless ssh; server side not in base image
RUN apt-get update && apt-get install -y openssh-server && rm -rf /var/lib/apt/lists/*

# PTCA env post-install: azureml-mlflow pulls cryptography (must be >=46.0.7 for CVE coverage;
# azureml-mlflow caps at <47.0.0). setuptools vendors jaraco.context (CVE-2026-23949) and
# also fixes wheel CVE-2026-24049; --force-reinstall --no-deps replaces the vendored copies.
RUN pip install --upgrade --no-cache-dir 'cryptography>=46.0.7'
RUN pip install --force-reinstall --no-deps 'setuptools>=82.0.1'

# torch 2.7.1+cu118: GHSA-887c-mr87-cxwp / CVE-2025-3730 (local DoS in ctc_loss; fixed in torch 2.8.0).
# Override NOT possible:
#   1. PyTorch upstream dropped CUDA 11.8 wheels starting with torch 2.8.0. The cu118 wheel index
#      (https://download.pytorch.org/whl/cu118/torch/) lists 2.7.1 as the highest stable release;
#      only a 2.8.0.dev* nightly exists for cu118 and PEP 440 dev versions still satisfy <2.8.0.
#   2. Installing the PyPI default torch 2.8.0 wheel (bundled cu126) would mismatch the rest of the
#      cu118-built GPU stack baked into the base image (DeepSpeed, onnxruntime-training-gpu,
#      torch-ort), breaking ABI / CUDA compatibility.
#   3. No cu118-patched base image is published (latest tag biweekly.202601.1 still ships torch 2.7.1+cu118).
# Permanent fix path: migrate workloads to acpt-pytorch-2.8-cuda12.6 (cu126 + torch 2.8.0).