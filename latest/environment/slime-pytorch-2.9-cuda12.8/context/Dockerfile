FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.8-ubuntu24.04:20260727.v1

USER root
# Editable sources live under /opt so AML/Singularity jobs (uid 9000) can
# read them. /root is mode 700 and would make `import slime`, the slime
# train.py entrypoint, and the Megatron-LM editable install unreachable
# from a non-root job user.
WORKDIR /opt

ARG SLIME_COMMIT=9b50665190d70cefcc9cc42e5994ad4de5f0cd88
ARG PATCH_VERSION=latest
ARG SGLANG_DEPENDENCY_VERSION=0.5.10.post1
ARG SGLANG_VERSION=0.5.11
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
ARG LOG4J_VERSION=2.25.4

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_PROGRESS_BAR=off
ENV PYTHONUNBUFFERED=1
ENV MAX_JOBS=1
ENV PYTHONPATH=/opt/Megatron-LM:${PYTHONPATH}

RUN set -eux; \
    find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
        -exec sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
            {} +; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        dnsutils \
        git \
        git-lfs \
        libnuma-dev \
        ninja-build \
        nvtop \
        openssh-client \
        openssh-server \
        rsync; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0

RUN python -m pip install --no-cache-dir \
        flash-linear-attention==0.4.1 \
        tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

RUN git clone https://github.com/THUDM/slime.git /opt/slime && \
    cd /opt/slime && \
    git checkout ${SLIME_COMMIT}

RUN NVCC_APPEND_FLAGS="--threads 1" \
    python -m pip install --disable-pip-version-check --no-cache-dir \
        --no-build-isolation \
        --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 1" \
        git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive /opt/Megatron-LM && \
    cd /opt/Megatron-LM && \
    git checkout ${MEGATRON_COMMIT} && \
    git apply /opt/slime/docker/patch/${PATCH_VERSION}/megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
        echo "Megatron patch failed to apply cleanly." && \
        exit 1; \
    fi && \
    python -m pip install --no-cache-dir -e .

RUN python -m pip install --no-cache-dir \
        git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT} --no-deps

RUN python -m pip install --no-cache-dir \
        git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 --force-reinstall && \
    python -m pip install --no-cache-dir \
        git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation && \
    python -m pip install --no-cache-dir \
        'nvidia-modelopt[torch]>=0.37.0' --no-build-isolation

# The PyTorch extension has no public wheel for this image's torch/CUDA stack.
# Install the hosted wheel extracted from the tested canary image instead of
# building transformer-engine-torch from source in the release pipeline.
RUN python -m pip install --no-cache-dir \
        https://github.com/yeshsurya/TransformerEngine/releases/download/transformer_engine_cu12-2.10.0-py3-none-manylinux_2_28_x86_64/transformer_engine_cu12-2.10.0-py3-none-manylinux_2_28_x86_64.whl && \
    python -m pip install --no-cache-dir \
        https://github.com/yeshsurya/TransformerEngine/releases/download/transformer_engine-2.10.0%2B769ed77-cp312-cp312-linux_x86_64/transformer_engine-2.10.0+769ed77-cp312-cp312-linux_x86_64.whl && \
    python -m pip install --no-cache-dir transformer-engine==2.10.0

# transformer-engine needs the FlashAttention backend for thd/packed (varlen)
# attention on A100 (sm80). Without flash-attn, TE finds no usable attention
# backend and aborts the first training forward with "No dot product attention
# backend is available for the provided inputs". cuDNN-fused does not cover this
# thd + padding_causal config on sm80, and the Unfused backend does not support
# thd, so flash-attn is required. No public wheel exists for this torch 2.9 /
# cu12.8 / cp312 stack, so install the hosted prebuilt wheel (TE 2.10 supports
# flash-attn <= 2.8.3) instead of building from source in the release pipeline.
RUN python -m pip install --no-cache-dir \
        https://github.com/yeshsurya/flash-attention/releases/download/flash_attn-2.8.3-cp312-cp312-linux_x86_64/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl

COPY requirements.txt /tmp/slime-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/slime-requirements.txt && \
    rm /tmp/slime-requirements.txt

# Bootstrap the validated dependency set, then upgrade only SGLang so its newer
# wheel does not replace this image's torch/CUDA stack.
RUN python -m pip install --no-cache-dir "sglang==${SGLANG_DEPENDENCY_VERSION}" && \
    python -m pip install --no-cache-dir --no-deps --upgrade "sglang==${SGLANG_VERSION}"
RUN python -m pip install --no-cache-dir \
        https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl \
        --force-reinstall

COPY patch_sglang_cve_2026_5760.py /tmp/patch_sglang_cve_2026_5760.py
RUN python /tmp/patch_sglang_cve_2026_5760.py && \
    rm /tmp/patch_sglang_cve_2026_5760.py

COPY patch_ray_log4j.py /tmp/patch_ray_log4j.py
# Ray vendors Log4j inside its fat jar, so overlay the fixed artifacts directly
# into the jar until Ray publishes a wheel with the patched dependency.
RUN LOG4J_VERSION=${LOG4J_VERSION} \
    LOG4J_API_SHA1=89ff2217b193fb187b134aa6ebcbfa8a28b018a9 \
    LOG4J_CORE_SHA1=b963c3d6bfdf05c61ad47a74e9f9295131607df2 \
    LOG4J_SLF4J_IMPL_SHA1=07c27f97ecedecf58341d4d4467bc3a58fbad73f \
    python /tmp/patch_ray_log4j.py && \
    rm /tmp/patch_ray_log4j.py

COPY patch_ray_jackson.py /tmp/patch_ray_jackson.py
# Ray vendors Jackson inside its fat jar; patch the embedded classes and Maven
# metadata so the embedded dependency is consistently upgraded.
RUN JACKSON_CORE_VERSION=2.21.5 \
    JACKSON_CORE_JAR_SHA1=7ec0c8bf8402fa998f64b2009632d6566a42ceb7 \
    JACKSON_CORE_POM_SHA1=32a98918e6a365962063ac88abe54b1251f1c2d0 \
    JACKSON_DATABIND_VERSION=2.21.5 \
    JACKSON_DATABIND_JAR_SHA1=84924990b468233a28214ae2c505a9ec24cd6ea6 \
    JACKSON_DATABIND_POM_SHA1=f85e75bea42ea3f61307a3dd9ba27ca33962be68 \
    python /tmp/patch_ray_jackson.py && \
    rm /tmp/patch_ray_jackson.py

RUN cd /opt/slime && \
    python -m pip install --no-cache-dir -e . --no-deps

# Apply Retail RFT compatibility patches: slime numpy/attention/GLM-4.6V fixes and
# neutralize transformer-engine 2.10's import-time PyPI-metadata / duplicate-.so
# checks (the meta package is a platlib wheel; the libraries themselves load fine).
COPY patch_slime_rft_fixes.py /tmp/patch_slime_rft_fixes.py
RUN python /tmp/patch_slime_rft_fixes.py && \
    rm /tmp/patch_slime_rft_fixes.py

RUN cd /opt/slime/slime/backends/megatron_utils/kernels/int4_qat && \
    python -m pip install --no-cache-dir . --no-build-isolation

# Upgrade vulnerable Ubuntu packages inherited from or installed by this image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --only-upgrade \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
        libarchive13t64 \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime \
        libpam0g \
        libsqlite3-0 \
        nginx \
        nginx-common \
        nginx-light \
        rsyslog \
        tar \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade vulnerable Python packages.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 && \
    python -m pip install --no-cache-dir \
        "aiohttp>=3.14.0" \
        "starlette>=1.0.1" \
        "idna>=3.15" \
        "pyarrow>=23.0.1" \
        "pyasn1>=0.6.4" \
        "pillow>=12.3.0" \
        "transformers>=5.5.0,<=5.6.0" \
        "mcp[cli]>=1.28.1" \
        "wandb>=0.28.1" \
        "GitPython>=3.1.55" \
        "cryptography>=49.0.0" \
        "setuptools>=83.0.0"

# The optional wandb service binary embeds vulnerable Go modules; remove it so
# runtime uses the Python client path instead.
RUN WANDB_CORE="$(python -c 'import pathlib, wandb; print(pathlib.Path(wandb.__file__).resolve().parent / "bin" / "wandb-core")')" && \
    test -f "${WANDB_CORE}" && \
    rm "${WANDB_CORE}"

COPY patch_ray_thirdparty.py /tmp/patch_ray_thirdparty.py
# Upgrade Ray's vendored Python packages and remove their obsolete metadata.
RUN python -m pip install --no-cache-dir \
        "idna>=3.15" \
        "aiohttp>=3.14.0" \
        --target "$(python -c 'import site; print(site.getsitepackages()[0])')/ray/_private/runtime_env/agent/thirdparty_files" && \
    python /tmp/patch_ray_thirdparty.py && \
    rm /tmp/patch_ray_thirdparty.py

# AML/Singularity jobs run as uid 9000 (aiscuser). Explicitly grant
# world read + traverse on the editable slime and Megatron-LM trees so
# `import slime`, `python /opt/slime/train.py`, and the PYTHONPATH-based
# Megatron-LM import all succeed without requiring `--user root`.
RUN chmod -R a+rX /opt/slime /opt/Megatron-LM
