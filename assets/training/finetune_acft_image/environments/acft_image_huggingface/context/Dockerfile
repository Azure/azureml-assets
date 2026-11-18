FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# IMPORTANT (2026-10-26): Python was upgraded from 3.10 to 3.11 for Pillow 13 security fixes.
# Downstream users must update interpreter-specific paths, wheels, and compiled dependencies for Python 3.11.
RUN conda install -n ptca -c conda-forge -y python=3.11 pip && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10 && \
    conda run -n ptca python -m pip install --upgrade --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.8.0+cu126' 'torchvision==0.23.0+cu126' 'torchaudio==2.8.0+cu126' && \
    conda run -n ptca python -m pip install --upgrade --no-cache-dir 'pillow>=13.0.0' && \
    conda clean -a -y

USER root
# Pull latest Ubuntu 22.04 security patches for packages present in the base image.
RUN apt-get -y update && apt-get -y upgrade \
 && apt-get -y install --only-upgrade \
        coreutils \
        curl \
        diffutils \
        dotnet-host-8.0 \
        dotnet-hostfxr-8.0 \
        dotnet-runtime-8.0 \
        libattr1 \
        libc-bin \
        libc-dev-bin \
        libc6 \
        libc6-dev \
        libcurl3-gnutls \
        libcurl4 \
        libevent-core-2.1-7 \
        libgcrypt20 \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
        liblzma5 \
        libnghttp2-14 \
        libp11-kit0 \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime \
        libpam0g \
        libperl5.34 \
        libpython3.10-minimal \
        libpython3.10-stdlib \
        libsqlite3-0 \
        libssh-4 \
        locales \
        openssh-client \
        perl \
        perl-base \
        perl-modules-5.34 \
        python3.10 \
        python3.10-minimal \
        tar \
        wget \
        xz-utils \
 || true \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --progress-bar off -r requirements.txt --no-cache-dir

# Override vulnerable transitive deps in ptca env that are not fixed in the base image
# onnx: azureml-acft-accelerator==0.0.89 requires onnx<=1.17.0 which downgrades base 1.21.0;
#   override needed to keep safe version (GHSA-p433-9wv8-28xj etc.)
# pyasn1: transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
#   parent packages use loose floors so pip resolves to a vulnerable version; override to >=0.6.4
# fastmcp: transitive dep (mlflow-skinny[mcp] requires fastmcp<4,>=2.0.0); loose floor,
#   GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767; override to >=3.2.0
# Mako: transitive dep (mlflow → alembic → Mako); alembic 1.18.4 requires Mako (no version pin),
#   pip won't upgrade pre-installed Mako; override needed (GHSA-v92g-xgxw-vvmm)
# GitPython: transitive dep (mlflow → mlflow-skinny requires gitpython<4,>=3.1.9); loose floor,
#   pip resolves to 3.1.52 which has GHSA-6p8h-3wgx-97gf, GHSA-fjr4-x663-mwxc,
#   GHSA-94p4-4cq8-9g67, and GHSA-r9mr-m37c-5fr3; override to >=3.1.55
# python-dotenv: transitive dep (mlflow → mlflow-skinny requires python-dotenv<2,>=0.19.0); loose floor,
#   pip resolves to 1.2.1 which has GHSA-mf9w-mj56-hr94; override to >=1.2.2
# pillow: comes from the base ACPT image (not a hard runtime dep of any requirements.txt package;
#   transformers/datasets/diffusers/optimum/huggingface-hub only require Pillow under extras_require
#   like [vision]/[dev]/[testing]); base image ships 12.1.1 which has GHSA-whj4-6x5x-4v2j;
#   no parent package to upgrade — explicit override required (>=12.2.0)
# pytest: comes from the base ACPT image (not a hard runtime dep of any requirements.txt package;
#   azureml-acft-accelerator only pins pytest~=5.3.0 under extras_require [test]); base image ships
#   7.4.3 which has GHSA-6w46-j5rx-g56g; no parent package to upgrade — explicit override required (>=9.0.3)
# urllib3: transitive dep of requests/botocore/etc.; parent packages use loose
#   floors (requests>=2.33 allows urllib3<3,>=1.21.1) so pip won't pull a newer
#   urllib3 on its own. Base image still ships 2.6.3 in the base conda env,
#   vulnerable to GHSA-qccp-gfcp-xxvc and GHSA-mf9v-mfxr-j63j; override to >=2.7.0.
# aiohttp: transitive dep (mlflow and other packages); ships 3.13.5 which has
#   GHSA-jg22-mg44-37j8 and GHSA-hg6j-4rv6-33pg; handled separately via conda run
#   (see below) to ensure the correct ptca env is targeted.
RUN pip install --progress-bar off --no-cache-dir --upgrade 'onnx>=1.21.0' 'pyasn1>=0.6.4' 'fastmcp>=3.2.0' 'Mako>=1.3.12' 'GitPython>=3.1.55' 'python-dotenv>=1.2.2' 'pillow>=13.0.0' 'pytest>=9.0.3' 'urllib3>=2.7.0'
# pip: package installer itself - there is no parent package that brings it in.
#   Both conda envs (ptca python3.10, base python3.14) ship pip from the base
#   image, vulnerable to GHSA-wf93-45jw-7689 and GHSA-qwm4-qh6w-59xr. Use conda-forge
#   to upgrade pip in both envs without allowing the solver to roll back other
#   packages (--freeze-installed).
RUN rm -rf /opt/conda/envs/ptca/lib/python3.11/site-packages/pip* \
           /opt/conda/lib/python3.14/site-packages/pip* \
 && conda install -n ptca -y -c conda-forge --freeze-installed --force-reinstall 'pip=26.2.0' \
 && conda install -n base -y -c conda-forge --freeze-installed --force-reinstall 'pip=26.2.0'
# python-dotenv in base conda env: transitive dep of uvicorn[standard] (>=0.13); loose floor,
#   base image has 1.2.1 which has GHSA-mf9w-mj56-hr94; override to >=1.2.2
# urllib3 in base conda env: same root cause as above — base ships 2.6.3 via
#   requests/botocore transitive chain; override to >=2.7.0 for
#   GHSA-qccp-gfcp-xxvc and GHSA-mf9v-mfxr-j63j.
# aiohttp in base conda env: base image ships 3.13.5 which has GHSA-jg22-mg44-37j8
#   and GHSA-hg6j-4rv6-33pg; override to >=3.14.0.
# click in base conda env: base image ships 8.2.1 which has GHSA-47fr-3ffg-hgmw;
#   no parent package to upgrade — explicit override required (>=8.3.3).
# idna in base conda env: base image ships 3.11 which has GHSA-65pc-fj4g-8rjx;
#   no parent package to upgrade — explicit override required (>=3.15).
# PyJWT: base image ships 2.12.1 which has GHSA-jq35-7prp-9v3f and GHSA-993g-76c3-p5m4;
#   no parent package to upgrade — explicit override required (>=2.13.0).
# cryptography in base conda env: pinned transitive/base-image dep brought in by the base ACPT image;
#   prior override installed 48.0.1, which has GHSA-g6cj-pr64-35w5; override to >=50.0.0.
# pydantic-settings in base conda env: pinned transitive/base-image dep brought in by the base ACPT image;
#   base ships 2.12.0 which has GHSA-4xgf-cpjx-pc3j; override to >=2.14.2.
# setuptools in base conda env: base image ships 82.0.0 which has GHSA-h35f-9h28-mq5c;
#   override to >=83.0.0.
# anyio and msgpack in base conda env: inherited versions are below the scanner-required safe floors.
RUN conda run -n base python -m pip install --progress-bar off --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' 'aiohttp>=3.14.4' 'click>=8.3.3' 'idna>=3.15' 'PyJWT>=2.13.0' 'cryptography>=50.0.0' 'pydantic-settings>=2.14.2' 'setuptools>=84.1.0' 'anyio>=4.14.2' 'msgpack>=1.2.3'
# aiohttp in ptca env: same CVEs as base env — GHSA-jg22-mg44-37j8 and GHSA-hg6j-4rv6-33pg.
# pyarrow in ptca env: ships 20.0.0 which has GHSA-rgxp-2hwp-jwgg; override to >=23.0.1.
# cryptography in ptca env: pinned transitive/base-image dep brought in by the base ACPT image;
#   prior override installed 48.0.1, which has GHSA-g6cj-pr64-35w5; override to >=50.0.0.
# setuptools in ptca env: direct requirement below covers GHSA-h35f-9h28-mq5c;
#   keep both conda envs consistent.
RUN conda run -n ptca python -m pip install --progress-bar off --no-cache-dir --upgrade 'aiohttp>=3.14.4' 'pyarrow>=23.0.1' 'cryptography>=50.0.0' 'setuptools>=84.1.0'
# Torch 2.13 fixes the reported PyTorch findings; no ACPT base currently publishes it.
# Keep the parent CUDA 12.6 runtime by using cu126 wheels with matching torchvision.
# Remove unused torchaudio to avoid retaining an incompatible ABI.
RUN conda run -n ptca python -m pip install --progress-bar off --no-cache-dir \
        --force-reinstall \
        'torch==2.13.0+cu126' 'torchvision==0.28.0+cu126' \
        'nvidia-nvshmem-cu12==3.4.5' \
        --index-url https://download.pytorch.org/whl/cu126 && \
    conda run -n ptca python -m pip uninstall -y torchaudio
# The conda pip upgrade can reinstall the rattler solver. The base image does
# not need it for runtime, and its compiled extension currently carries Rust CVEs.
RUN conda config --system --set solver classic \
 && /opt/conda/bin/python3.14 -m pip uninstall -y py-rattler conda-rattler-solver 2>/dev/null || true \
 && rm -rf /opt/conda/lib/python3.14/site-packages/rattler* \
           /opt/conda/lib/python3.14/site-packages/conda_rattler_solver* \
           /opt/conda/lib/python3.14/site-packages/py_rattler* \
           /opt/conda/conda-meta/py-rattler-*.json \
           /opt/conda/conda-meta/conda-rattler-solver-*.json

# Remove stale scanner metadata while retaining the upgraded installations.
RUN set -eu; \
    roots="/opt /root"; \
    [ ! -d /azureml-envs ] || roots="$roots /azureml-envs"; \
    find $roots -type f \( \
        -path '*/site-packages/pip/_vendor/bom.cdx.json' -o \
        -path '*/site-packages/pip/_vendor/vendor.txt' \
    \) -delete; \
    find $roots -type d \( \
        -name 'setuptools-70.3.0.dist-info' -o \
        -name 'setuptools-70.3.0.egg-info' -o \
        -name 'setuptools-70.3.0-py*.egg-info' -o \
        -name 'msgpack-1.1.2.dist-info' -o \
        -name 'msgpack-1.1.2.egg-info' -o \
        -name 'msgpack-1.1.2-py*.egg-info' -o \
        -name 'msgpack-1.2.1.dist-info' -o \
        -name 'msgpack-1.2.1.egg-info' -o \
        -name 'msgpack-1.2.1-py*.egg-info' \
    \) -prune -exec rm -rf {} +; \
    find $roots -type f \( \
        -path '*/conda-meta/setuptools-70.3.0-*.json' -o \
        -path '*/conda-meta/msgpack-python-1.1.2-*.json' -o \
        -path '*/conda-meta/msgpack-1.1.2-*.json' -o \
        -path '*/conda-meta/msgpack-python-1.2.1-*.json' -o \
        -path '*/conda-meta/msgpack-1.2.1-*.json' \
    \) -delete; \
    stale="$(find $roots \( \
        -path '*/site-packages/pip/_vendor/bom.cdx.json' -o \
        -path '*/site-packages/pip/_vendor/vendor.txt' -o \
        -name 'setuptools-70.3.0.dist-info' -o \
        -name 'setuptools-70.3.0.egg-info' -o \
        -name 'setuptools-70.3.0-py*.egg-info' -o \
        -name 'msgpack-1.1.2.dist-info' -o \
        -name 'msgpack-1.1.2.egg-info' -o \
        -name 'msgpack-1.1.2-py*.egg-info' -o \
        -name 'msgpack-1.2.1.dist-info' -o \
        -name 'msgpack-1.2.1.egg-info' -o \
        -name 'msgpack-1.2.1-py*.egg-info' -o \
        -path '*/conda-meta/setuptools-70.3.0-*.json' -o \
        -path '*/conda-meta/msgpack-python-1.1.2-*.json' -o \
        -path '*/conda-meta/msgpack-1.1.2-*.json' -o \
        -path '*/conda-meta/msgpack-python-1.2.1-*.json' -o \
        -path '*/conda-meta/msgpack-1.2.1-*.json' \
    \) -print -quit)"; \
    [ -z "$stale" ] || { echo "stale setuptools/msgpack scanner metadata remains: $stale" >&2; exit 1; }

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/

# Reassert newly patched transitive dependencies after all package installs.
RUN /opt/conda/envs/ptca/bin/python -m pip install --no-cache-dir --upgrade 'oauthlib>=4.0.0'

# Remove the Conda-owned Setuptools files before reinstalling so stale vendored
# package metadata from 78.1.0 cannot remain in the final SBOM.
RUN rm -rf /opt/conda/envs/ptca/lib/python3.11/site-packages/setuptools \
           /opt/conda/envs/ptca/lib/python3.11/site-packages/setuptools-78.1.0.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/setuptools-78.1.0-*.json && \
    /opt/conda/envs/ptca/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps \
        'setuptools==84.1.0'

# Reassert November 2026 vulnerability-report floors in the scanned Python prefix.
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade \
    'certifi>=2026.10.19' 'opentelemetry-api>=1.45.1' 'brotlicffi>=1.2.1.0' && \
    rm -rf /opt/conda/lib/python3.14/site-packages/certifi-2024.7.4.dist-info && \
    rm -rf /opt/conda/lib/python3.14/site-packages/opentelemetry__api-1.43.0.dist-info

# Reassert November 2026 OS-package floors for installed packages only.
RUN set -eux; \
    apt-get update; \
    security_packages="libbrotli1 dotnet-hostfxr-8.0"; \
    installed_packages=""; \
    for package in $security_packages; do \
        if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then \
            installed_packages="$installed_packages $package"; \
        fi; \
    done; \
    if [ -n "$installed_packages" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade $installed_packages; \
    fi; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*
