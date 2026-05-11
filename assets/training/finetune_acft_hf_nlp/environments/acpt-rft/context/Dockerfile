FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade --fix-missing || \
    (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y upgrade) && \
    apt-get autoremove -y && \
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
# wandb 0.26.1 ships wandb-core compiled with Go 1.26.1 (needs 1.26.2); no fixed wandb release yet.
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
# vllm pinned to 0.19.1 to fix GHSA-6c4r-fmh3-7rh8 (CVE in librosa transitive dep).
# Root-cause analysis: librosa was vendored via vllm's `audio` extra; vllm PR #37058 removed
# librosa entirely. PyPI metadata confirms vllm 0.18.0 still lists `librosa; extra == "audio"`
# while 0.18.1+ (incl. 0.19.1) do NOT. 0.19.1 also fixes CVE-2026-7141.
# Parent package (verl 0.7.0) constrains `vllm<=0.12.0,>=0.8.5` only with the [vllm] extra,
# which is not used here; verl is installed without the extra, so we override vllm directly.
# Staying on the 0.19.x line (same torch==2.10.0 ABI as 0.19.0) preserves compatibility with
# the pinned flash-attn wheel and the verl/vLLM internal API patches in vllm_async_server,
# vllm_rollout, and utils. 0.20.x bumps torch to 2.11.0 and was avoided.
RUN pip install vllm==0.19.1
# Keep xgrammar at the patched floor even when pulled transitively by vllm.
RUN pip install --no-cache-dir 'xgrammar>=0.1.32'
RUN pip install openai==2.14.0
RUN pip install --force-reinstall --no-cache-dir --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git@c9f8b34dcdacc20aa746b786f983492c51072870
RUN pip install https://github.com/yeshsurya/flash-attention/releases/download/v2.8.3-linux-1/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl
# Fix security vulnerabilities in ptca conda env not resolved by base image
# (pip, setuptools, wheel, aiohttp, protobuf, requests, onnx, pytest are already at
#  safe versions in base image biweekly.202605.1 and do not need overrides)
# cryptography==46.0.7: CVE-2026-41727; not pre-installed in ptca env, pulled by azureml-mlflow
# fastmcp>=3.2.0: GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767
# Mako>=1.3.11: CVE-2025-46803; transitive dep of alembic, parent uses loose floor
# lxml>=6.1.0: GHSA-vfmq-68hx-4jfw; transitive dep of multiple packages, parent uses loose floor
# transformers>=5.0.0rc3: GHSA-69w3-r845-3855 (CVE-2026-1839); direct dep, upgraded to patched 5.x
# GitPython>=3.1.47: GHSA-x2qx-6953-8485, GHSA-rpm5-65cw-6hj4; transitive dep of wandb (requires
#   gitpython!=3.1.29,>=1.0.0 as of 0.26.1), parent uses loose floor — no wandb release forces >=3.1.47
RUN pip install --upgrade cryptography==46.0.7 'fastmcp>=3.2.0' 'Mako>=1.3.11' 'lxml>=6.1.0' 'transformers>=5.0.0rc3' 'GitPython>=3.1.47'
# python-dotenv>=1.2.2: GHSA-mf9w-mj56-hr94; transitive dep of pydantic-settings (requires >=0.21.0),
#   uvicorn (optional, requires >=0.13), and fastmcp (requires >=1.1.0). All parents use loose floors,
#   so no parent upgrade can force >=1.2.2. Base image ships 1.2.1 in base conda env; we patch
#   both base (python 3.13) and ptca (python 3.10) envs to cover all install paths.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
RUN pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
# ray vendors its own copy of aiohttp inside thirdparty_files/ for runtime_env agent;
# the vendored copy is not upgraded by pip install above. Patching all copies in-place.
RUN find /opt/conda/envs/ptca/lib/python3.10/site-packages/ray -type d -name 'thirdparty_files' | while read dir; do \
      rm -rf "$dir"/aiohttp*; \
      pip install --no-cache-dir --target "$dir" 'aiohttp>=3.13.4'; \
    done
COPY vllm_rollout /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_rollout.py
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/ /opt/conda/pkgs/

