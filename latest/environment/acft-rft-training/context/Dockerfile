FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202603.2
USER root
# Security: upgrade all OS packages and remove stale kernel headers to patch USN vulnerabilities
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*.deb
RUN conda install pip==25.3 -y
# Remove all old pip metadata after upgrade
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-acft-common-components==0.0.88
RUN pip install azureml-evaluate-mlflow==0.0.88
RUN pip install verl==0.7.0
RUN pip install sacrebleu==2.5.1
COPY tracking /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py

RUN pip install --no-cache-dir accelerate==1.10.0
RUN pip install --no-cache-dir sglang==0.5.4
RUN pip install --no-cache-dir sgl-kernel==0.3.16.post3

RUN pip uninstall -y mlflow 
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
RUN pip install --no-cache-dir starlette==0.49.1
RUN pip install --no-cache-dir cryptography==46.0.5
# Upgrade wandb to fix golang.org/x/crypto vulnerabilities (CVE-2025-47914, CVE-2025-58181)
RUN pip install --no-cache-dir --upgrade "wandb>=0.23.0"
RUN pip install --no-cache-dir triton==3.4.0
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
COPY vllm_async_server /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py
COPY __init__ /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/__init__.py
COPY azure_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_grader.py
COPY azure_python_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_python_grader.py
COPY utils /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/vllm/utils.py
RUN python3 -m pip install --upgrade pip==26.0 setuptools==82.0.0 wheel==0.46.2
RUN pip install vllm==0.17.0
# xgrammar 0.1.29 → ≥0.1.32 (CVE-2026-25048 — segfault via nested syntax)
#   vllm pins xgrammar==0.1.29; all vllm versions (up to 0.17.1) use this exact pin,
#   so no vllm upgrade can fix this — explicit override after vllm install is required.
RUN pip install --no-cache-dir 'xgrammar>=0.1.32'
RUN pip install openai==2.14.0
RUN pip install --force-reinstall --no-cache-dir --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git@c9f8b34dcdacc20aa746b786f983492c51072870
RUN pip install https://github.com/yeshsurya/flash-attention/releases/download/v2.8.3-linux-1/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl
# aiohttp: vllm/sglang use loose floors, pip can't force 3.13.3 transitively
# protobuf: vllm uses loose floors (>=6.30.0), pip can't force 6.33.5 transitively
RUN pip install --upgrade aiohttp==3.13.3 protobuf==6.33.5 setuptools==82.0.0 pip==26.0 wheel==0.46.2 cryptography==46.0.5
# Fix vulnerabilities in base conda env (python3.13) from ACPT base image (biweekly.202603.1)
# Still vulnerable in base: cryptography(44.0.1), pip(25.3), setuptools(80.9.0), wheel(0.45.1)
RUN conda run -n base pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.5
COPY vllm_rollout /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_rollout.py
# Clean up pip caches and old package files to prevent vulnerability detection
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -y -n ptca pip>=26.0.1 wheel>=0.46.2
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/

