FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.8-ubuntu24.04:{{latest-image-tag}}

USER root
WORKDIR /root

ARG SLIME_COMMIT=9b50665190d70cefcc9cc42e5994ad4de5f0cd88
ARG PATCH_VERSION=latest
ARG SGLANG_VERSION=0.5.10.post1
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
ARG LOG4J_VERSION=2.25.4

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV MAX_JOBS=1
ENV PYTHONPATH=/root/Megatron-LM:${PYTHONPATH}

RUN set -eux; \
    find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
        -exec sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
            {} +; \
    for attempt in 1 2 3 4 5; do \
        apt-get update -o Acquire::Retries=5 -o APT::Update::Error-Mode=any && break; \
        if [ "$attempt" = "5" ]; then exit 1; fi; \
        sleep $((attempt * 15)); \
    done; \
    apt-get -y -o Acquire::Retries=5 upgrade; \
    apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
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

RUN python -m pip install --upgrade \
        pip==26.0 \
        setuptools==82.0.1 \
        wheel==0.46.2 \
        packaging

RUN python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 \
        torchvision==0.24.1 \
        torchaudio==2.9.1

RUN python -m pip install \
        flash-linear-attention==0.4.1 \
        tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

RUN git clone https://github.com/THUDM/slime.git /root/slime && \
    cd /root/slime && \
    git checkout ${SLIME_COMMIT}

RUN NVCC_APPEND_FLAGS="--threads 1" \
    python -m pip install --disable-pip-version-check --no-cache-dir \
        --no-build-isolation \
        --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 1" \
        git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive /root/Megatron-LM && \
    cd /root/Megatron-LM && \
    git checkout ${MEGATRON_COMMIT} && \
    git apply /root/slime/docker/patch/${PATCH_VERSION}/megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
        echo "Megatron patch failed to apply cleanly." && \
        exit 1; \
    fi && \
    python -m pip install -e .

RUN python -m pip install \
        git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT} --no-deps

RUN python -m pip install \
        git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 --no-cache-dir --force-reinstall && \
    python -m pip install \
        git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation && \
    python -m pip install \
        'nvidia-modelopt[torch]>=0.37.0' --no-build-isolation

COPY requirements.txt /tmp/slime-requirements.txt
RUN python -m pip install --ignore-installed PyJWT && \
    python -m pip install -r /tmp/slime-requirements.txt

RUN python -m pip install "sglang==${SGLANG_VERSION}"
RUN python -m pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall

RUN python -m pip install "numpy<2" \
        'pillow>=12.2.0' \
        'cryptography>=46.0.5' \
        'protobuf>=6.33.5' \
        'urllib3>=2.6.3' \
        'requests>=2.33.0' \
        'aiohttp>=3.13.4' \
        'filelock>=3.20.3' \
        'PyJWT>=2.12.0'

COPY patch_ray_log4j.py /tmp/patch_ray_log4j.py
RUN LOG4J_VERSION=${LOG4J_VERSION} \
    LOG4J_API_SHA1=89ff2217b193fb187b134aa6ebcbfa8a28b018a9 \
    LOG4J_CORE_SHA1=b963c3d6bfdf05c61ad47a74e9f9295131607df2 \
    LOG4J_SLF4J_IMPL_SHA1=07c27f97ecedecf58341d4d4467bc3a58fbad73f \
    python /tmp/patch_ray_log4j.py && \
    rm /tmp/patch_ray_log4j.py

RUN cd /root/slime && \
    python -m pip install -e . --no-deps

RUN cd /root/slime/slime/backends/megatron_utils/kernels/int4_qat && \
    python -m pip install . --no-build-isolation

COPY smoke_test.py /tmp/smoke_test.py
RUN python /tmp/smoke_test.py && \
    rm /tmp/smoke_test.py

RUN rm -rf /root/.cache/pip /tmp/slime-requirements.txt
