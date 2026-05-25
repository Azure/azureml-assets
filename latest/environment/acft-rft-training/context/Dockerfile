FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.2
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade --fix-missing || \
    (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y upgrade) && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*.deb
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-acft-common-components==0.0.91
RUN pip install azureml-evaluate-mlflow==0.0.91
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
# SURGICAL FIX (CVE-2026-44223 / GHSA-83vm-p52w-f9pw / VCM 5012192):
#   The upstream fix (vllm-project/vllm#38610) is a single-line change in
#   vllm/v1/spec_decode/extract_hidden_states.py: `return sampled_token_ids` →
#   `return sampled_token_ids[:, :1]`. We backport this as a file overlay on the
#   installed vllm 0.19.1 (see `COPY extract_hidden_states ...` near the end of
#   this Dockerfile + the build-time verification that asserts the overlay
#   landed and vllm.__version__ remains 0.19.1). This pattern matches the
#   existing overlays for vllm_async_server / vllm_rollout / utils.
#
# RESIDUAL FINDINGS (accepted risk, not remediated in this build):
#   - GHSA-hpv8-x276-m59f / VCM 5012004 / CVE-2026-44222 (vLLM multimodal
#     special-token placeholder DoS via OpenAI-compatible API server) — fixed
#     in vllm>=0.20.0. The upstream fix (umbrella issue vllm-project/vllm#32656)
#     is a broad multimodal refactor across per-model files (qwen2_vl,
#     qwen2_5_vl, qwen2_5_omni_thinker, qwen3_vl, ernie45_vl, glm4v, keye…) that
#     cannot be backported as a small overlay without significant code-shift
#     risk. We accept this finding because the RFT image consumes vLLM only as
#     an in-process rollout backend for trusted training workloads — it does
#     not expose vLLM's OpenAI-compatible multimodal endpoint to unauthenticated
#     callers, so the remote DoS vector is unreachable in this deployment.
#   - GHSA-83vm-p52w-f9pw / VCM 5012192 — see SURGICAL FIX above (REMEDIATED).
# We are NOT upgrading to vllm 0.20.x in this build. Cascade blockers (re-verified
# against PyPI requires_dist and Dao-AILab/yeshsurya GitHub release assets on 2026-05-23):
#   1. torch ABI pin: vllm 0.20.0/0.20.1/0.20.2/0.21.0 all require torch==2.11.0 exact
#      (PyPI requires_dist). Current vllm 0.19.1 resolves torch==2.10.0; sglang 0.5.10
#      pins torch==2.9.1. The minimum sglang line accepting torch 2.11.0 is sglang 0.5.11
#      (also bumps transformers==5.6.0 + new sgl-kernel/torch-memory-saver/flashinfer matrix).
#   2. flash-attn wheel — THIS IS THE HARD BLOCKER. The image installs a prebuilt cp310
#      flash-attn wheel from yeshsurya/flash-attention; only v2.8.3-linux-1 is published
#      and it is built against the torch 2.10 ABI. The upstream Dao-AILab flash-attention
#      release feed (latest fa4-v4.0.0.beta14, 2026-05-23) publishes no cp310 wheels at
#      all (assets list empty), so there is no torch-2.11 / cp310 wheel available to
#      consume. Building flash-attn from source inside ACR exceeds the build timeout.
#   3. vLLM v1-engine internal patches: the COPY'd files (vllm_async_server, vllm_rollout,
#      utils) import `vllm.v1.engine.async_llm.AsyncLLM`, `vllm.v1.engine.core.EngineCoreProc`,
#      `vllm.v1.engine.utils.CoreEngineProcManager`, `vllm.v1.executor.abstract.Executor`,
#      `vllm.utils.argparse_utils`, `vllm.utils.network_utils`, `vllm.config.LoRAConfig`. These
#      v1-engine internals frequently shift across vllm minor lines (0.19→0.20) and would
#      require a full re-validation of the patches.
#   (Previously-listed blocker #4, verl 0.7.0 + transformers>=5.6.0 ImportError on
#    AutoModelForVision2Seq, is RESOLVED upstream in verl 0.7.1 — the import is now
#    wrapped in try/except per github.com/volcengine/verl@v0.7.1/verl/utils/model.py
#    line 1. We can adopt verl 0.7.1 if/when blockers #1/#2 are unblocked; not bumped
#    here to keep this security change minimal.)
# Risk acceptance: this image consumes vLLM internally for RFT training rollouts; it is
# deployed in internal/trusted training workloads and does not expose a public OpenAI
# endpoint for unauthenticated multimodal traffic, so the practical exposure of the DoS path
# is limited. The override avoids a high-risk torch / sglang / flash-attn / DeepGEMM /
# custom-vLLM-patch requalification in a single security bump. Sister env acpt-grpo runs
# vllm==0.20.1 successfully BUT does NOT pin sglang/flash-attn/torch/verl and so does not
# hit this cascade.
# NOTE on SBOM visibility: the CVE-2026-44223 overlay (extract_hidden_states.py, see
# end of file) closes the runtime vulnerability but does NOT update vllm's dist-info
# METADATA Version field, so SBOM-based scanners (Qualys/VCM) will continue to report
# CVE-2026-44223 and CVE-2026-44222 against vllm@0.19.1 until the cascade can be lifted.
RUN pip install vllm==0.19.1
# Keep xgrammar at the patched floor even when pulled transitively by vllm.
RUN pip install --no-cache-dir 'xgrammar>=0.1.32'
RUN pip install openai==2.14.0
RUN pip install --force-reinstall --no-cache-dir --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git@c9f8b34dcdacc20aa746b786f983492c51072870
RUN pip install https://github.com/yeshsurya/flash-attention/releases/download/v2.8.3-linux-1/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl
# Security overrides for pip-installed packages whose parent packages do not pin them safely.
# cryptography==46.0.7: CVE-2026-41727; not pre-installed in ptca env, pulled by azureml-mlflow
# fastmcp>=3.2.0: GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767
# Mako>=1.3.11: CVE-2025-46803; transitive dep of alembic, parent uses loose floor
# lxml>=6.1.0: GHSA-vfmq-68hx-4jfw; transitive dep of multiple packages, parent uses loose floor
# transformers>=5.0.0rc3: GHSA-69w3-r845-3855 (CVE-2026-1839); direct dep. No upper cap because
#   the actual installed transformers ends up at 5.8.x (pulled forward by vllm 0.19.1
#   requires_dist `transformers>=4.56.0` followed by later installs) — verl 0.7.0 imports
#   `AutoModelForVision2Seq` at module top level, but `verl.utils.model` is not imported at
#   build/verification time in this image, so the symbol absence in transformers 5.6+ does
#   not affect the build. (verl 0.7.1 wraps that import in try/except; see vllm cascade
#   comment above for full context.)
# GitPython>=3.1.47: GHSA-x2qx-6953-8485, GHSA-rpm5-65cw-6hj4; transitive dep of wandb (requires
#   gitpython!=3.1.29,>=1.0.0 as of 0.26.1), parent uses loose floor — no wandb release forces >=3.1.47
# pyOpenSSL>=26.0.0: CVE-2026-27459 (HIGH, DTLS cookie callback buffer overflow) and
#   CVE-2026-27448 (LOW, TLS connection bypass via unhandled callback exception). Base image
#   ships pyOpenSSL 25.3.0; azureml-core 1.61.0.post3 pins pyopenssl<26.0.0 and no newer
#   azureml-core release exists (verified 2026-05-23), so explicit override is required.
#   Pattern matches sister env acpt-grpo.
RUN pip install --upgrade cryptography==46.0.7 'fastmcp>=3.2.0' 'Mako>=1.3.11' 'lxml>=6.1.0' 'transformers>=5.0.0rc3' 'GitPython>=3.1.47' 'pyOpenSSL>=26.0.0'
# Base env (py3.13) and ptca env (py3.10) overrides for packages where every parent pins a loose floor.
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
#   (verified via PyPI requires_dist on 2026-05-23; matches sister env acpt-grpo).
# idna>=3.15 (base env, py3.13; also patched inside ray vendored thirdparty_files below):
#   GHSA-65pc-fj4g-8rjx / VCM 5012909 (CVE-2026-45409, CRITICAL). idna is pulled transitively
#   by requests/urllib3/cryptography/httpx/anyio/etc., none of which pin idna>=3.15 in any
#   currently published release (verified via PyPI requires_dist on 2026-05-23 — all parents
#   use loose floors like `idna>=2.5` or `idna<4`). Direct override is the only remediation.
# click>=8.3.3 (base env, py3.13): GHSA-47fr-3ffg-hgmw / VCM 5012984 (CVE-2026-7246, HIGH,
#   click.edit() command injection). click is bootstrapped into the base conda env and pulled
#   by typer/uvicorn/black/flask/etc.; none of these pin click>=8.3.3 in published releases
#   (verified PyPI requires_dist 2026-05-23). Direct override is the only remediation.
#   (Note: requirements.txt also pins click==8.3.3 to cover the ptca env install path.)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'pip>=26.1.1' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json
RUN pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'pip>=26.1.1' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json
# ray vendors its own copies of aiohttp and idna inside thirdparty_files/ for the runtime_env
# agent; those vendored copies are not upgraded by the pip installs above. Patching all copies
# in-place (aiohttp>=3.13.4 closes prior CVE; idna>=3.15 closes CVE-2026-45409 / VCM 5012909).
RUN find /opt/conda/envs/ptca/lib/python3.10/site-packages/ray -type d -name 'thirdparty_files' | while read dir; do \
      rm -rf "$dir"/aiohttp* "$dir"/idna*; \
      pip install --no-cache-dir --target "$dir" 'aiohttp>=3.13.4' 'idna>=3.15'; \
    done
COPY vllm_rollout /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_rollout.py
# CVE-2026-44223 surgical backport: overlay the patched extract_hidden_states.py
# (verbatim copy of vllm v0.19.1 with the one-line PR #38610 fix applied).
# Must come AFTER every step that could pip-touch vllm; the verification RUN
# below fails the build fast if the overlay did not land or vllm version drifted.
COPY extract_hidden_states /opt/conda/envs/ptca/lib/python3.10/site-packages/vllm/v1/spec_decode/extract_hidden_states.py
RUN find /opt/conda/envs/ptca/lib/python3.10/site-packages/vllm -name '*.pyc' -delete && \
    /opt/conda/envs/ptca/bin/python -c "import vllm; assert vllm.__version__ == '0.19.1', vllm.__version__; \
import vllm.v1.spec_decode.extract_hidden_states as m, inspect; src = inspect.getsource(m); \
assert 'sampled_token_ids[:, :1]' in src, 'CVE-2026-44223 overlay missing'; \
assert 'CVE-2026-44223' in src, 'CVE-2026-44223 marker missing'; \
print('vllm', vllm.__version__, 'CVE-2026-44223 overlay verified')"
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/ /opt/conda/pkgs/

