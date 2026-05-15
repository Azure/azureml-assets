FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.8-ubuntu24.04:20260514.v1

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
        torch==2.9.1 \
        torchvision==0.24.1 \
        torchaudio==2.9.1

RUN python -m pip install --no-cache-dir \
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
    python -m pip install --no-cache-dir -e .

RUN python -m pip install --no-cache-dir \
        git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT} --no-deps

RUN python -m pip install --no-cache-dir \
        git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 --force-reinstall && \
    python -m pip install --no-cache-dir \
        git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation && \
    python -m pip install --no-cache-dir \
        'nvidia-modelopt[torch]>=0.37.0' --no-build-isolation

COPY requirements.txt /tmp/slime-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/slime-requirements.txt && \
    rm /tmp/slime-requirements.txt

RUN python -m pip install --no-cache-dir "sglang==${SGLANG_VERSION}"
RUN python -m pip install --no-cache-dir \
        https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl \
        --force-reinstall

COPY patch_ray_log4j.py /tmp/patch_ray_log4j.py
# Ray vendors Log4j inside ray_dist.jar as expanded fat-jar contents plus
# Maven metadata. Installing newer Log4j jars beside Ray would not change the
# classes or package metadata that runtime and vulnerability scanners observe,
# so overlay the fixed Log4j artifacts directly into Ray's fat jar until Ray
# publishes a wheel with the patched dependency.
RUN LOG4J_VERSION=${LOG4J_VERSION} \
    LOG4J_API_SHA1=89ff2217b193fb187b134aa6ebcbfa8a28b018a9 \
    LOG4J_CORE_SHA1=b963c3d6bfdf05c61ad47a74e9f9295131607df2 \
    LOG4J_SLF4J_IMPL_SHA1=07c27f97ecedecf58341d4d4467bc3a58fbad73f \
    python /tmp/patch_ray_log4j.py && \
    rm /tmp/patch_ray_log4j.py

RUN cd /root/slime && \
    python -m pip install --no-cache-dir -e . --no-deps

RUN cd /root/slime/slime/backends/megatron_utils/kernels/int4_qat && \
    python -m pip install --no-cache-dir . --no-build-isolation

COPY smoke_test.py /tmp/smoke_test.py
RUN python /tmp/smoke_test.py && \
    rm /tmp/smoke_test.py
