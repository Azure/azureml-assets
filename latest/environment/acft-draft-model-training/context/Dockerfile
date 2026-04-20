#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202604.2
USER root
# Security: upgrade all OS packages, pin linux headers to patched versions, and upgrade binutils/git/wget
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --only-upgrade linux-headers-5.15.0-163-generic=5.15.0-163.173 linux-headers-5.15.0-163=5.15.0-163.173 linux-libc-dev=5.15.0-163.173 || \
    apt-get install -y --only-upgrade linux-headers-generic linux-libc-dev && \
    apt-get install -y --only-upgrade git tar binutils binutils-common binutils-x86-64-linux-gnu libbinutils wget && \
    apt-get autoremove -y linux-headers-5.15.0-153 linux-headers-5.15.0-153-generic linux-headers-5.15.0-161 linux-headers-5.15.0-161-generic 2>/dev/null && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Security: upgrade base conda env (python3.13) from ACPT base image (biweekly.202603.1)
# Still vulnerable: cryptography(44.0.1), pip(25.3), setuptools(80.9.0), wheel(0.45.1)
RUN conda run -n base python -m pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.7 'aiohttp>=3.13.4' 'requests>=2.33.0'
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-acft-common-components==0.0.89
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow==0.0.89

# following are for vulnerability overrides at later\
# release of following packages consider moving then to requirements.txt
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
RUN pip install --no-cache-dir --upgrade "wandb>=0.23.0"
# GHSA-7rgv-gqhr-fxg3: xgrammar stack exhaustion DoS; patched in 0.1.32
RUN pip install xgrammar==0.1.32
RUN pip install transformers==4.57.1
# upgrade pip, wheel, setuptools and transitive deps to fix vulnerabilities
# protobuf: wandb/google-cloud-storage cap <7, override needed
# cryptography: azureml-mlflow pins <46.0.0; override needed for CVE fix
# aiohttp: transitive dep of ray/vllm/azure-core; parents use loose floors (GHSA-mwh4-6h8g-pg8w etc.)
# onnx: onnxruntime accepts onnx>=1.16.0; override needed (GHSA-p433-9wv8-28xj etc.)
# fastmcp: GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767; >=3.2.0 required
# anthropic: GHSA-q5f5-3gjm-7mfm, GHSA-w828-4qhx-vxx3; >=0.87.0 required
# requests: transitive dep of azure-core/mlflow/transformers; parents use loose floors (GHSA-gc5v-m9x4-r6x2)
RUN pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.7 protobuf==6.33.5 \
    'aiohttp>=3.13.4' 'requests>=2.33.0' 'onnx>=1.21.0' 'fastmcp>=3.2.0' 'anthropic>=0.87.0'
# Fix vulnerabilities in the ptca conda environment (pre-built in base image, not targeted by above installs)
# CVE-2026-1703 (pip), CVE-2026-24049 (wheel)
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade "pip>=26.0" "wheel>=0.46.2" && \
    rm -f /opt/conda/envs/ptca/conda-meta/wheel-0.45.1*.json /opt/conda/envs/ptca/conda-meta/pip-25.3*.json
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
