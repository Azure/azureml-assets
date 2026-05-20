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
# vllm pinned to 0.19.1 to fix:
#   - GHSA-6c4r-fmh3-7rh8 (librosa transitive dep removed in vllm 0.18.1+ via PR #37058;
#     PyPI metadata confirms 0.18.0 still lists `librosa; extra == "audio"` while 0.18.1+ do not)
#   - CVE-2026-7141 (fixed in 0.19.1)
#   - GHSA-x368-4g9h-fvv4 / VCM 5012008 (fix landed in 0.19.1)
# Parent package (verl 0.7.0) constrains `vllm<=0.12.0,>=0.8.5` only via the optional [vllm]
# extra, which is NOT used in this image (verl is installed without the extra); thus there is
# no parent that pulls vllm — it is a direct top-level install here, and the only available
# remediation path is a direct version override.
#
# RESIDUAL FINDINGS (not remediated in this build):
#   - GHSA-hpv8-x276-m59f / VCM 5012004 (multimodal token-injection DoS in vLLM's
#     OpenAI-compatible API server) — fixed in vllm>=0.20.0.
#   - GHSA-83vm-p52w-f9pw / VCM 5012192 — fixed in vllm>=0.20.0.
#   - CVE-2026-44222 (vLLM special-token placeholder DoS) — fixed in vllm>=0.20.0.
#   - CVE-2026-44223 (vLLM extract_hidden_states speculative-decoding crash) — fixed in vllm>=0.20.0.
# We are NOT upgrading to 0.20.x in this build because the cascade has four concrete
# blockers verified via PyPI metadata and an ACR build attempt on 2026-05-20:
#   1. sglang stack: vllm 0.20.0 requires torch==2.11.0 (exact pin, verified via
#      https://pypi.org/pypi/vllm/0.20.0/json `requires_dist`); the currently pinned
#      sglang==0.5.10 requires torch==2.9.1 (also exact). The minimum sglang line that allows
#      torch 2.11.0 is sglang==0.5.11 (also bumps transformers==5.6.0 and pulls a new
#      sgl-kernel/torch-memory-saver matrix) — a multi-package transition.
#   2. flash-attn ABI: the prebuilt wheel
#      https://github.com/yeshsurya/flash-attention/releases/download/v2.8.3-linux-1/
#      flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl is the only asset published at that
#      release tag and is built against an older torch ABI (torch 2.10 era, matching the
#      torch that vllm 0.19.x resolves to); no torch 2.11 build is published there.
#   3. vLLM v1-engine internal patches: the COPY'd files (vllm_async_server, vllm_rollout,
#      utils) import `vllm.v1.engine.async_llm.AsyncLLM`, `vllm.v1.engine.core.EngineCoreProc`,
#      `vllm.v1.engine.utils.CoreEngineProcManager`, `vllm.v1.executor.abstract.Executor`,
#      `vllm.utils.argparse_utils`, `vllm.utils.network_utils`, `vllm.config.LoRAConfig`. These
#      v1-engine internals frequently shift across vllm minor lines (0.19→0.20) and would
#      require a full re-validation of the patches.
#   4. verl 0.7.0 + transformers 5.6.0 incompatibility (empirically observed on ACR run ca96,
#      2026-05-20): transformers 5.6.0 (pulled by sglang 0.5.11) removed `AutoModelForVision2Seq`,
#      which verl 0.7.0/verl.utils.model imports at top level → ImportError on module load.
#      Upgrading transformers therefore requires a verl bump as well, multiplying scope.
# Risk acceptance: this image consumes vLLM internally for RFT training rollouts; it is
# deployed in internal/trusted training workloads and does not expose a public OpenAI
# endpoint for unauthenticated multimodal traffic, so the practical exposure of the DoS path
# is limited. The override avoids a high-risk torch / sglang / flash-attn / DeepGEMM /
# custom-vLLM-patch requalification in a single security bump. Re-evaluate in the next
# refresh once the flash-attn wheel, the vllm_async_server/vllm_rollout patches, and verl
# are all updated for vllm 0.20.x + transformers 5.6 (sister env acpt-grpo already runs
# vllm==0.20.1 successfully, but acpt-grpo does NOT pin sglang/flash-attn/torch/verl and so
# does not hit this cascade).
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
# pyOpenSSL>=26.0.0: CVE-2026-27459 (HIGH, DTLS cookie callback buffer overflow) and
#   CVE-2026-27448 (LOW, TLS connection bypass via unhandled callback exception). Base image
#   ships pyOpenSSL 25.3.0; azureml-core 1.61.0.post3 pins pyopenssl<26.0.0 and no newer
#   azureml-core release exists (verified 2026-05-20), so explicit override is required.
#   Pattern matches sister env acpt-grpo.
RUN pip install --upgrade cryptography==46.0.7 'fastmcp>=3.2.0' 'Mako>=1.3.11' 'lxml>=6.1.0' 'transformers>=5.0.0rc3' 'GitPython>=3.1.47' 'pyOpenSSL>=26.0.0'
# python-dotenv>=1.2.2: GHSA-mf9w-mj56-hr94; transitive dep of pydantic-settings (requires >=0.21.0),
#   uvicorn (optional, requires >=0.13), and fastmcp (requires >=1.1.0). All parents use loose floors,
#   so no parent upgrade can force >=1.2.2. Base image ships 1.2.1 in base conda env; we patch
#   both base (python 3.13) and ptca (python 3.10) envs to cover all install paths.
# pip>=26.1.1: GHSA-jp4c-xjxw-mgf9 / VCM 5011855 (CVE-2026-6357). Base image biweekly.202605.x
#   ships pip 26.0.1 in BOTH the ptca (py3.10) and base (py3.13) conda envs (per scan paths).
#   pip is the Python package installer itself — it is bootstrapped by the conda/python
#   distribution and has no parent package that pulls it in, so the only available remediation
#   is a direct upgrade in each conda environment. Pattern matches sister env acpt-grpo.
#   NOTE: `pip install --upgrade pip` replaces the on-disk pip files but does NOT update
#   conda's metadata DB at conda-meta/pip-*.json. SBOM scanners that read conda-meta still
#   report the old pip version, so we explicitly delete the stale pip-26.0*.json files.
# urllib3>=2.7.0 (base env, py3.13): GHSA-mf9v-mfxr-j63j / VCM 5012484 and
#   GHSA-qccp-gfcp-xxvc / VCM 5012480. urllib3 is brought in transitively in the base env by
#   requests/botocore/azureml-core/kubernetes/etc.; all of these only constrain urllib3<3
#   (loose), so no parent upgrade forces >=2.7.0. Direct override is the only remediation
#   (verified via PyPI requires_dist on 2026-05-19; matches sister env acpt-grpo).
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'pip>=26.1.1' 'urllib3>=2.7.0' && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json
RUN pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'pip>=26.1.1' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json
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

