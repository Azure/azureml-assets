FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade --fix-missing || \
    (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y upgrade) && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*.deb
# USN-8298-1: dotnet 8.0.27; USN-8319-1: libgcrypt20 1.9.4-3ubuntu3.2
# These packages are not always picked up by apt-get upgrade (held back), so install explicitly.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade \
        dotnet-hostfxr-8.0 dotnet-host-8.0 dotnet-runtime-8.0 \
        libgcrypt20 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install verl==0.7.1
RUN pip install sacrebleu==2.5.1
COPY tracking /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py

RUN pip install --no-cache-dir accelerate==1.10.0
RUN pip install --no-cache-dir sglang==0.5.11
RUN pip install --no-cache-dir sglang-kernel==0.4.2

RUN pip uninstall -y mlflow 
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
# Upgrade wandb to latest; remove wandb-core Go binary to fix GO-2026-4864..4947
# wandb 0.26.1 ships wandb-core compiled with Go 1.26.1 (needs 1.26.2); no fixed wandb release yet.
# Removing the binary forces wandb to use its Python backend (safe fallback).
RUN pip install --no-cache-dir --upgrade "wandb>=0.26.0" && \
    find /opt/conda/envs/ptca -name 'wandb-core' -path '*/wandb/bin/*' -delete 2>/dev/null || true
RUN pip install --no-cache-dir --upgrade torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
COPY vllm_async_server /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py
COPY __init__ /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/__init__.py
COPY azure_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_grader.py
COPY azure_python_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_python_grader.py
COPY utils /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/vllm/utils.py
# vllm is a direct top-level runtime dependency in this image. PyPI metadata checks show
# verl only declares vllm under the unused [vllm] extra, so there is no parent package to
# upgrade for GHSA-hpv8-x276-m59f / GHSA-83vm-p52w-f9pw.
RUN pip install --no-cache-dir vllm==0.20.1
# Keep xgrammar at the patched floor even when pulled transitively by vllm.
RUN pip install --no-cache-dir 'xgrammar>=0.1.32'
RUN pip install openai==2.14.0
RUN pip install --force-reinstall --no-cache-dir --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git@c9f8b34dcdacc20aa746b786f983492c51072870
RUN pip install --no-cache-dir 'flash-attn-4>=4.0.0b9'
# Security overrides for pip-installed packages whose parent packages do not pin them safely.
# cryptography==46.0.7: CVE-2026-41727; not pre-installed in ptca env, pulled by azureml-mlflow
# fastmcp>=3.2.0: GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767
# Mako>=1.3.11: CVE-2025-46803; transitive dep of alembic, parent uses loose floor
# lxml>=6.1.0: GHSA-vfmq-68hx-4jfw; transitive dep of multiple packages, parent uses loose floor
# transformers==5.6.0: GHSA-69w3-r845-3855 (CVE-2026-1839); direct dep and exact sglang
#   0.5.11 requirement. verl 0.7.1 wraps the older AutoModelForVision2Seq import.
# GitPython>=3.1.47: GHSA-x2qx-6953-8485, GHSA-rpm5-65cw-6hj4; transitive dep of wandb (requires
#   gitpython!=3.1.29,>=1.0.0 as of 0.26.1), parent uses loose floor — no wandb release forces >=3.1.47
# pyOpenSSL>=26.0.0: CVE-2026-27459 and CVE-2026-27448. azureml-core 1.61.0.post3 still
#   pins pyopenssl<26.0.0, so explicit override is required.
# starlette>=1.0.1: GHSA-86qp-5c8j-p5mr; placed here (after vllm) so that vllm's transitive
#   fastapi dependency cannot downgrade starlette to the vulnerable 0.x series.
RUN pip install --upgrade cryptography==46.0.7 'fastmcp>=3.2.0' 'Mako>=1.3.11' 'lxml>=6.1.0' 'transformers==5.6.0' 'GitPython>=3.1.47' 'pyOpenSSL>=26.0.0' 'starlette>=1.0.1'
# Base env (py3.13) and ptca env (py3.10) overrides for packages where every parent pins a loose floor.
# python-dotenv>=1.2.2: GHSA-mf9w-mj56-hr94; transitive dep of pydantic-settings, uvicorn,
#   and fastmcp. All parents use loose floors, so no parent upgrade can force >=1.2.2.
# pip>=26.1.1: GHSA-jp4c-xjxw-mgf9 / VCM 5011855 (CVE-2026-6357). pip is bootstrapped by
#   conda/python and has no parent package that can be upgraded, so patch both conda envs.
#   NOTE: `pip install --upgrade pip` replaces the on-disk pip files but does NOT update
#   conda's metadata DB at conda-meta/pip-*.json. SBOM scanners that read conda-meta still
#   report the old pip version, so we explicitly delete stale pip-26.0*.json files.
# urllib3>=2.7.0 (base env, py3.13): GHSA-mf9v-mfxr-j63j / VCM 5012484 and
#   GHSA-qccp-gfcp-xxvc / VCM 5012480. urllib3 is brought in transitively by
#   requests/botocore/azureml-core/kubernetes/etc.; all of these only constrain urllib3<3
#   (loose), so no parent upgrade forces >=2.7.0. Direct override is the only remediation
#   (verified via PyPI requires_dist on 2026-05-23; matches sister env acpt-grpo).
# idna>=3.15 (base env, py3.13; also patched inside ray vendored thirdparty_files below):
#   GHSA-65pc-fj4g-8rjx / VCM 5012909 (CVE-2026-45409, CRITICAL). idna is pulled transitively
#   by requests/urllib3/cryptography/httpx/anyio/etc., none of which pin idna>=3.15 in any
#   currently published release (verified via PyPI requires_dist on 2026-05-23 — all parents
#   use loose floors like `idna>=2.5` or `idna<4`). Direct override is the only remediation.
# click>=8.3.3 (base env, py3.13): GHSA-47fr-3ffg-hgmw / VCM 5012984 (CVE-2026-7246, HIGH,
#   click.edit() command injection). click is bootstrapped into the conda env and pulled
#   by typer/uvicorn/black/flask/etc.; none of these pin click>=8.3.3 in published releases
#   (verified PyPI requires_dist 2026-05-23). Direct override is the only remediation.
#   (Note: requirements.txt also pins click==8.3.3 to cover the ptca env install path.)
# PyJWT>=2.13.0 (base env, py3.13): GHSA-993g-76c3-p5m4, GHSA-jq35-7prp-9v3f / VCM 5013533,
#   5013521. PyJWT is pulled transitively by various auth/azure packages; no parent pins
#   >=2.13.0 in published releases. Direct override is the only remediation.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'pip>=26.1.1' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'PyJWT>=2.13.0' && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json
# GHSA-65pc-fj4g-8rjx: idna>=3.15 for ptca env (base env covered above)
# GHSA-993g-76c3-p5m4, GHSA-jq35-7prp-9v3f: PyJWT>=2.13.0 for ptca env (base env covered above)
RUN pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'pip>=26.1.1' 'idna>=3.15' 'PyJWT>=2.13.0' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json
# ray vendors its own copies of aiohttp and idna inside thirdparty_files/ for the runtime_env
# agent; those vendored copies are not upgraded by the pip installs above. Patching all copies
# in-place (aiohttp>=3.14.0 closes GHSA-hg6j-4rv6-33pg/GHSA-jg22-mg44-37j8; idna>=3.15 closes CVE-2026-45409 / VCM 5012909).
# Also upgrade aiohttp in base (py3.13) and ptca (py3.10) envs to >=3.14.0.
# GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8: aiohttp vulnerabilities; patched in >=3.14.0.
RUN conda run -n base python -m pip install --no-cache-dir 'aiohttp>=3.14.0' && \
    pip install --no-cache-dir 'aiohttp>=3.14.0'
RUN find /opt/conda/envs/ptca/lib/python3.10/site-packages/ray -type d -name 'thirdparty_files' | while read dir; do \
      rm -rf "$dir"/aiohttp* "$dir"/idna*; \
      pip install --no-cache-dir --target "$dir" 'aiohttp>=3.14.0' 'idna>=3.15'; \
    done
COPY vllm_rollout /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_rollout.py
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/ /opt/conda/pkgs/

