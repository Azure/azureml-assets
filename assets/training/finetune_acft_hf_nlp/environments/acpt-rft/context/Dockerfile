FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root
# Security: upgrade all OS packages and patch USN-8176-1 (.NET 8.0 vulnerabilities)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade \
        dotnet-host-8.0 dotnet-hostfxr-8.0 dotnet-runtime-8.0 2>/dev/null || true && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*.deb
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install verl==0.7.0
RUN pip install sacrebleu==2.5.1
COPY tracking /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py

RUN pip install --no-cache-dir accelerate==1.10.0
RUN pip install --no-cache-dir sglang==0.5.10
RUN pip install --no-cache-dir sgl-kernel==0.3.16.post3

RUN pip uninstall -y mlflow 
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
RUN pip install --no-cache-dir starlette==0.49.1
# Upgrade wandb to latest; remove wandb-core Go binary to fix GO-2026-4864..4947
# wandb 0.26.0 ships wandb-core compiled with Go 1.26.1 (needs 1.26.2); no fixed wandb release yet.
# Removing the binary forces wandb to use its Python backend (safe fallback).
RUN pip install --no-cache-dir --upgrade "wandb>=0.26.0" && \
    find /opt/conda/envs/ptca -name 'wandb-core' -path '*/wandb/bin/*' -delete 2>/dev/null || true
RUN pip install --no-cache-dir triton==3.4.0
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
COPY vllm_async_server /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py
COPY __init__ /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/__init__.py
COPY azure_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_grader.py
COPY azure_python_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_python_grader.py
COPY utils /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/vllm/utils.py
RUN python3 -m pip install --upgrade pip==26.0 setuptools==82.0.0 wheel==0.46.2
RUN pip install vllm==0.19.0
# Keep xgrammar at the patched floor even when pulled transitively by vllm.
RUN pip install --no-cache-dir 'xgrammar>=0.1.32'
RUN pip install openai==2.14.0
RUN pip install --force-reinstall --no-cache-dir --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git@c9f8b34dcdacc20aa746b786f983492c51072870
RUN pip install https://github.com/yeshsurya/flash-attention/releases/download/v2.8.3-linux-1/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl
# Fix security vulnerabilities in ptca conda env (active environment)
# aiohttp==3.13.4: GHSA-63hf-3vf5-4wqf et al. (8 CVEs); transitive dep of vllm/sglang/azure SDK,
#   parents use loose floors — pip can't force >=3.13.4 transitively
# protobuf==6.33.5: CVE-2026-40186; transitive dep of vllm (requires >=6.30.0), parent uses loose floor
# setuptools==82.0.0: GHSA-58pv-8j8x-9vj2 (vendored jaraco.context), GHSA-8rrh-rw8j-w5fx (vendored wheel)
# pip==26.0: GHSA-4xh5-x5gv-qwph, GHSA-6vgw-5pg2-w6jp
# wheel==0.46.2: CVE-2026-24049
# cryptography==46.0.7: CVE-2026-41727; azureml-mlflow pins <46.0.0 but upgrading anyway for CVE fix
# requests>=2.33.0: GHSA-gc5v-m9x4-r6x2; transitive dep of azure-core/mlflow/transformers,
#   msal requires >=2.0.0,<3, azureml-core requires >=2.19.1,<3 — no parent pins safe floor
# onnx>=1.21.0: GHSA-p433-9wv8-28xj, GHSA-538c-55jv-c5g9 et al.; transitive dep of onnxruntime/
#   azureml-acft-accelerator (require >=1.16.0), no parent upgrade resolves this
# fastmcp>=3.2.0: GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767
# Mako>=1.3.11: CVE-2025-46803; transitive dep of alembic, parent uses loose floor
# pytest>=9.0.3: GHSA-6w46-j5rx-g56g; pre-installed in ptca env from base image, no parent to upgrade
# lxml>=6.1.0: GHSA-vfmq-68hx-4jfw; transitive dep of multiple packages, parent uses loose floor
# transformers: GHSA-69w3-r845-3855 requires >=5.0.0rc3 but no such version exists on PyPI
#   (latest stable is 4.57.6); cannot fix until upstream publishes a patched release
RUN pip install --upgrade aiohttp==3.13.4 protobuf==6.33.5 setuptools==82.0.0 pip==26.0 wheel==0.46.2 cryptography==46.0.7 'requests>=2.33.0' 'onnx>=1.21.0' 'fastmcp>=3.2.0' 'Mako>=1.3.11' 'pytest>=9.0.3' 'lxml>=6.1.0'
# Fix vulnerabilities in base conda env (python 3.13) from ACPT base image (biweekly.202603.1)
# Base env ships: cryptography(44.0.1), pip(25.3), setuptools(80.9.0), wheel(0.45.1), aiohttp(3.12.x)
# Same CVEs as ptca env above; base env is separate and must be patched independently
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.7 aiohttp==3.13.4
# ray vendors its own copy of aiohttp inside thirdparty_files/ for runtime_env agent;
# the vendored copy is not upgraded by pip install above. Patching all copies in-place.
RUN find /opt/conda/envs/ptca/lib/python3.10/site-packages/ray -type d -name 'thirdparty_files' | while read dir; do \
      rm -rf "$dir"/aiohttp*; \
      pip install --no-cache-dir --target "$dir" 'aiohttp==3.13.4'; \
    done
COPY vllm_rollout /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_rollout.py
# Clean up pip caches and old package files to prevent vulnerability detection
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
# Set secure defaults
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -y -n ptca 'pip>=26.0.1' 'wheel>=0.46.2'
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/ /opt/conda/pkgs/

