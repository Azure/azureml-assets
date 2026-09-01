FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
ARG VLLM_WHEEL=https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade --fix-missing || \
    (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y upgrade) && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*.deb
# Security packages are not always picked up by apt-get upgrade, so install them explicitly.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade \
        dotnet-hostfxr-8.0 dotnet-host-8.0 dotnet-runtime-8.0 \
        libgcrypt20 \
        tar libsqlite3-0 wget \
        libkrb5support0 libgssapi-krb5-2 libk5crypto3 libkrb5-3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .

RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}} azureml-metrics[all]=={{latest-pypi-version}}
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install verl==0.7.1
RUN pip install sacrebleu==2.5.1
COPY tracking /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py

RUN pip install --no-cache-dir accelerate==1.10.0
RUN pip install --no-cache-dir sglang==0.5.11
RUN pip install --no-cache-dir sglang-kernel==0.4.2

RUN pip uninstall -y mlflow
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
# Upgrade wandb to latest; remove wandb-core Go binary to fix GO-2026-4864..4947.
# wandb 0.26.1 ships wandb-core compiled with Go 1.26.1 (needs 1.26.2); no fixed wandb release yet.
# Removing the binary forces wandb to use its Python backend (safe fallback).
RUN pip install --no-cache-dir --upgrade "wandb>=0.26.0" && \
    find /opt/conda/envs/ptca -name 'wandb-core' -path '*/wandb/bin/*' -delete 2>/dev/null || true
COPY vllm_async_server /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py
COPY __init__ /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/__init__.py
COPY azure_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_grader.py
COPY azure_python_grader /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/azure_python_grader.py
COPY utils /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/vllm/utils.py
# vLLM's default PyPI wheel is built for CUDA 13. Install the official CUDA 12.9
# wheel so A100 v4 hosts with a CUDA 12.8 driver can load vllm._C.
RUN pip install --no-cache-dir "${VLLM_WHEEL}"
# Keep native Torch packages on the CUDA lineage vLLM 0.26 was built against.
# torchaudio is unused by this image and must not remain inherited at a
# potentially incompatible ABI.
RUN pip install --no-cache-dir --force-reinstall \
        torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu129 && \
    pip uninstall -y torchaudio
# vllm's resolver can upgrade numpy, but verl 0.7.1 requires numpy<2.
RUN pip install --no-cache-dir numpy==1.26.4
# Keep xgrammar at the patched floor when pulled transitively by vllm; fixes GHSA-7rgv-gqhr-fxg3.
RUN pip install --no-cache-dir 'xgrammar>=0.1.32'
RUN pip install openai==2.25.0
RUN pip install --force-reinstall --no-cache-dir --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git@c9f8b34dcdacc20aa746b786f983492c51072870
RUN pip install --no-cache-dir 'flash-attn-4>=4.0.0b9'
# Security overrides for pip-installed packages whose parent packages do not pin them safely.
# cryptography>=48.0.1: GHSA-537c-gmf6-5ccf; transitive dep pulled by azureml-mlflow
# pydantic-settings>=2.14.2: GHSA-4xgf-cpjx-pc3j; transitive dep pulled by fastmcp
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
# sqlparse>=0.6.0: pinned transitive dep of mlflow/azureml-mlflow; fixes GHSA-f2ff-p2ww-7p4p,
#   GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, and GHSA-pwgv-4x5q-6m9f.
RUN pip install --upgrade 'cryptography>=48.0.1' 'pydantic-settings>=2.14.2' 'fastmcp>=3.2.0' 'Mako>=1.3.11' 'lxml>=6.1.0' 'transformers==5.6.0' 'GitPython>=3.1.47' 'pyOpenSSL>=26.0.0' 'starlette>=1.0.1' 'sqlparse>=0.6.0'
# Base env (py3.13) and ptca env (py3.10) overrides for packages where every parent pins a loose floor.
# cryptography>=48.0.1 (base env, py3.13): GHSA-537c-gmf6-5ccf; direct override
#   keeps the base env patched if later package resolution leaves an older copy.
# python-dotenv>=1.2.2: GHSA-mf9w-mj56-hr94; transitive dep of pydantic-settings, uvicorn,
#   and fastmcp. All parents use loose floors, so no parent upgrade can force >=1.2.2.
# pip>=26.1.2: GHSA-wf93-45jw-7689 / VCM 5015223. pip is bootstrapped by
#   conda/python and has no parent package that can be upgraded, so patch both conda envs.
#   NOTE: `pip install --upgrade pip` replaces the on-disk pip files but does NOT update
#   conda's metadata DB at conda-meta/pip-*.json. SBOM scanners that read conda-meta still
#   report the old pip version, so we explicitly delete stale pip metadata files.
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
# msgpack>=1.2.1 (base env, py3.13): GHSA-6v7p-g79w-8964 / VCM 5014304.
#   msgpack is pulled transitively by distributed runtime packages with loose floors.
# setuptools>=83.0.0 (base env, py3.14): GHSA-h35f-9h28-mq5c / VCM 5016030.
#   setuptools is bootstrapped by the base conda image, so patch it directly.
# Remove obsolete vendored dependency metadata after upgrading pip.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'cryptography>=48.0.1' 'python-dotenv>=1.2.2' 'pip>=26.1.2' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'PyJWT>=2.13.0' 'msgpack>=1.2.1' 'setuptools>=83.0.0' && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json /opt/conda/conda-meta/pip-26.1.1*.json \
          /opt/conda/conda-meta/setuptools-70*.json /opt/conda/conda-meta/setuptools-81*.json /opt/conda/conda-meta/setuptools-82*.json \
          /opt/conda/conda-meta/msgpack-python-1.1*.json /opt/conda/conda-meta/msgpack-1.1*.json && \
    rm -f /opt/conda/lib/python3.14/site-packages/pip/_vendor/bom.cdx.json && \
    sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==83.0.0/' \
          /opt/conda/lib/python3.14/site-packages/pip/_vendor/vendor.txt
# GHSA-65pc-fj4g-8rjx: idna>=3.15 for ptca env (base env covered above)
# GHSA-993g-76c3-p5m4, GHSA-jq35-7prp-9v3f: PyJWT>=2.13.0 for ptca env (base env covered above)
# GHSA-h35f-9h28-mq5c: setuptools>=83.0.0 for ptca env (base env covered above)
RUN pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'pip>=26.1.2' 'idna>=3.15' 'PyJWT>=2.13.0' 'setuptools>=83.0.0' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json /opt/conda/envs/ptca/conda-meta/pip-26.1.1*.json \
          /opt/conda/envs/ptca/conda-meta/setuptools-70*.json /opt/conda/envs/ptca/conda-meta/setuptools-81*.json /opt/conda/envs/ptca/conda-meta/setuptools-82*.json \
          /opt/conda/envs/ptca/conda-meta/msgpack-python-1.1*.json /opt/conda/envs/ptca/conda-meta/msgpack-1.1*.json && \
        rm -f /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/bom.cdx.json && \
    sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==83.0.0/' \
            /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/vendor.txt
# Ray vendors its own copies of aiohttp and idna inside thirdparty_files/ for the runtime_env
# agent; those vendored copies are not upgraded by the pip installs above.
# aiohttp>=3.14.3 closes GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, and GHSA-mfx4-hv73-q22v.
RUN conda run -n base python -m pip install --no-cache-dir 'aiohttp>=3.14.3' && \
    pip install --no-cache-dir 'aiohttp>=3.14.3'
RUN find /opt/conda/envs/ptca/lib/python3.10/site-packages/ray -type d -name 'thirdparty_files' | while read dir; do \
      rm -rf "$dir"/aiohttp* "$dir"/idna*; \
      pip install --no-cache-dir --target "$dir" 'aiohttp>=3.14.3' 'idna>=3.15'; \
    done
# Ray's distribution jar shades vulnerable Maven artifacts that pip cannot upgrade.
# Patch jackson-databind/core, vendored by ray, for GHSA-5jmj-h7xm-6q6v,
# GHSA-5gvw-p9qm-jgwh, GHSA-mhm7-754m-9p8w, and GHSA-r7wm-3cxj-wff9; patch
# log4j-api/core for CVE-2026-49844, GHSA-445c-vh5m-36rj,
# GHSA-6hg6-v5c8-fphq, and GHSA-3pxv-7cmr-fjr4; patch httpcore5 for
# GHSA-hf6x-8p5f-cgmf.
COPY patch_ray_jackson_databind.py /tmp/patch_ray_jackson_databind.py
RUN JACKSON_DATABIND_VERSION=2.21.5 \
    JACKSON_DATABIND_JAR_SHA1=84924990b468233a28214ae2c505a9ec24cd6ea6 \
    JACKSON_DATABIND_POM_SHA1=f85e75bea42ea3f61307a3dd9ba27ca33962be68 \
    JACKSON_CORE_VERSION=2.18.8 \
    JACKSON_CORE_JAR_SHA1=459eb2c4bb30bd6898eb9bbe9d7a4c4140529b8b \
    JACKSON_CORE_POM_SHA1=90813471a8fd344b176d7459ab2ee310dacb607f \
    LOG4J_API_VERSION=2.25.5 \
    LOG4J_API_JAR_SHA1=6c80ea65e41f7858f7e15cb390c98c36167c7b3a \
    LOG4J_API_POM_SHA1=214797155b00469cc53ffb33ca76055f0b2bc755 \
    LOG4J_CORE_VERSION=2.25.5 \
    LOG4J_CORE_JAR_SHA1=c194a13e18b0a36014e867d2d323b989bf212151 \
    LOG4J_CORE_POM_SHA1=1f76ca57995fe0e588dc92ff3aa9f26d45af79ba \
    HTTPCORE5_VERSION=5.4.3 \
    HTTPCORE5_JAR_SHA1=32a0699fbcb103f5621c00983ee8f5aac080e4a6 \
    HTTPCORE5_POM_SHA1=4553084e7ef07928bc1c2b65d4b099b972ba53a6 \
    /opt/conda/envs/ptca/bin/python /tmp/patch_ray_jackson_databind.py && \
    rm -f /tmp/patch_ray_jackson_databind.py
COPY vllm_rollout /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/workers/rollout/vllm_rollout/vllm_rollout.py
# ---- verl RFT (GRPO) compatibility patches ----------------------------------
# Reconciles the bundled kernels / transformers / vLLM / verl / flash-attn lineages
# so the verl Reinforced Fine-Tuning (GRPO) recipe runs end-to-end (model load,
# FSDP2, async vLLM agentic rollout, validation, GRPO update). The script applies a
# small set of idempotent, anchor-verified source patches and fails the build if any
# target anchor is missing. Run in the training conda env.
COPY verl_rft_compat_patches /tmp/verl_rft_compat_patches.py
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir "accelerate==1.14.0" && \
/opt/conda/envs/ptca/bin/pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu129 && \
/opt/conda/envs/ptca/bin/pip install --no-cache-dir --force-reinstall --no-deps \
    flashinfer-python==0.6.13 flashinfer-cubin==0.6.13 && \
/opt/conda/envs/ptca/bin/pip uninstall -y torchcodec nvidia-nccl-cu13 && \
/opt/conda/envs/ptca/bin/pip install --no-cache-dir --force-reinstall --no-deps \
    nvidia-nccl-cu12==2.29.7 && \
    /opt/conda/envs/ptca/bin/python /tmp/verl_rft_compat_patches.py && \
    rm -f /tmp/verl_rft_compat_patches.py
COPY cuda_preflight.py /opt/conda/envs/ptca/bin/acft-rft-cuda-preflight
RUN chmod +x /opt/conda/envs/ptca/bin/acft-rft-cuda-preflight && \
    /opt/conda/envs/ptca/bin/acft-rft-cuda-preflight --skip-cuda
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/*
ENV PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/ /opt/conda/pkgs/
