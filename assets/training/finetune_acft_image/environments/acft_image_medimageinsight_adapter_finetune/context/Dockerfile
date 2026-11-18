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

RUN apt-get -y update && apt-get -y upgrade \
 && apt-get -y install --only-upgrade \
        coreutils \
        curl \
        diffutils \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        libattr1 \
        libc6 \
        libc6-dev \
        libc-bin \
        libc-dev-bin \
        libevent-core-2.1-7 \
        libgssapi-krb5-2 \
        libgcrypt20 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
        libcurl3-gnutls \
        libcurl4 \
        liblzma5 \
        libp11-kit0 \
        libperl5.34 \
        libpng16-16 \
        libpython3.10-minimal \
        libpython3.10-stdlib \
        libsqlite3-0 \
        libssh-4 \
        libssl-dev \
        libssl3 \
        locales \
        openssh-client \
        openssl \
        perl \
        perl-base \
        perl-modules-5.34 \
        python3.10 \
        python3.10-minimal \
        tar \
        wget \
        xz-utils \
 && apt-get -y install --no-install-recommends unzip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required PyPI packages.
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir && \
    rm -rf /opt/conda/envs/ptca/lib/python3.11/site-packages/setuptools-70.3.0.dist-info \
           /opt/conda/envs/ptca/lib/python3.11/site-packages/msgpack-1.1*.dist-info

# Torch 2.13 fixes the reported PyTorch findings; no ACPT base currently publishes it.
# Keep the parent CUDA 12.6 runtime by using cu126 wheels with matching torchvision.
# Keep torchaudio at the latest compatible cu126 release available.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.13.0' \
        'torchvision==0.28.0' \
        'torchaudio==2.11.0'

# Base-env transitive dependency overrides. The base image is patched for some
# of these, but this image's dependency layer can resolve AzureML/MLflow
# transitive dependencies below the fixed floors.
# urllib3 2.6.3 is pulled in by requests 2.33.1 (`urllib3<3,>=1.26`);
# requests 2.34.2 still does not require urllib3>=2.7.0.
# idna 3.11 is required by requests 2.33.1, httpx 0.28.1, anyio 4.12.1, and
# yarl 1.23.0; their parent metadata still does not require idna>=3.15.
# click 8.2.1 is required by anaconda-cli-base 0.8.2 and typer 0.25.1; parent
# metadata only requires `click` or `click>=8.2.1`, not click>=8.3.3.
# python-dotenv 1.2.1 is required by anaconda-auth 0.14.4 and
# pydantic-settings; parents still use unpinned python-dotenv or
# python-dotenv>=0.21.0.
RUN conda install -y -n base -c conda-forge 'urllib3=2.7.0' 'idna=3.15' 'click=8.3.3' 'python-dotenv=1.2.2' && \
    rm -rf /opt/conda/lib/python3.14/site-packages/urllib3-2.6*.dist-info \
           /opt/conda/lib/python3.14/site-packages/idna-3.11.dist-info \
           /opt/conda/lib/python3.14/site-packages/click-8.2*.dist-info \
           /opt/conda/lib/python3.14/site-packages/python_dotenv-1.2.1.dist-info \
           /opt/conda/lib/python3.14/site-packages/python-dotenv-1.2.1.dist-info && \
    conda clean -ay

# The conda-forge conda update can add optional conda-rattler-solver/py-rattler
# metadata. This image does not need that solver plugin after build-time package
# installation, so remove it to avoid carrying unused solver artifacts.
RUN /opt/conda/bin/pip uninstall -y py-rattler conda-rattler-solver && \
    rm -rf /opt/conda/lib/python3.14/site-packages/rattler \
           /opt/conda/lib/python3.14/site-packages/py_rattler* \
           /opt/conda/lib/python3.14/site-packages/py-rattler* \
           /opt/conda/lib/python3.14/site-packages/conda_rattler_solver* && \
    rm -f /opt/conda/conda-meta/py-rattler-*.json \
          /opt/conda/conda-meta/conda-rattler-solver-*.json

# Upgrade aiohttp to patched version in both envs (GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'aiohttp>=3.14.4' && \
    conda run -n ptca python -m pip install --no-cache-dir --upgrade 'aiohttp>=3.14.4'

# pyarrow: pinned transitive dep in ptca; brought in by AzureML dependencies, fixes GHSA-rgxp-2hwp-jwgg.
# cryptography: pinned transitive dep of ACPT base image; brought in by base image, fixes GHSA-g6cj-pr64-35w5.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade 'pyarrow>=23.0.1' 'cryptography==50.0.0' && \
    rm -rf /opt/conda/envs/ptca/lib/python3.11/site-packages/cryptography-48.0.1.dist-info

# setuptools: direct dependency in requirements.txt and inherited from ACPT base image; fixes CVE-2026-59890.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'setuptools==84.1.0' && \
    conda run -n ptca python -m pip install --no-cache-dir --upgrade 'setuptools==84.1.0' && \
    rm -rf /opt/conda/lib/python3.14/site-packages/setuptools-8[12]*.dist-info \
           /opt/conda/envs/ptca/lib/python3.11/site-packages/setuptools-8[12]*.dist-info

# PyJWT: pinned transitive dep in base; brought in by base image, fixes GHSA-jq35-7prp-9v3f and GHSA-993g-76c3-p5m4.
# cryptography: pinned transitive dep of ACPT base image; brought in by base image, fixes GHSA-g6cj-pr64-35w5.
# pydantic-settings: pinned transitive dep of azureml-defaults -> azureml-inference-server-http, fixes GHSA-4xgf-cpjx-pc3j.
# msgpack: pinned transitive dep of azureml-mlflow -> mlflow-skinny, fixes GHSA-6v7p-g79w-8964.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'PyJWT>=2.13.0' 'cryptography==50.0.0' 'pydantic-settings>=2.14.2' 'msgpack>=1.2.3' && \
    rm -rf /opt/conda/lib/python3.14/site-packages/cryptography-48.0.1.dist-info \
           /opt/conda/lib/python3.14/site-packages/pydantic_settings-2.12*.dist-info \
           /opt/conda/lib/python3.14/site-packages/pydantic__settings-2.12*.dist-info \
           /opt/conda/lib/python3.14/site-packages/msgpack-1.1*.dist-info \
           /opt/conda/lib/python3.14/site-packages/setuptools-70.3.0.dist-info

# pip: direct package manager upgraded in both envs; fixes GHSA-qwm4-qh6w-59xr.
# anyio: pinned transitive dep of anaconda-cli-base/httpx tooling in base env; brought in by base image, fixes GHSA-5p39-cfhj-2xmp and GHSA-82r6-8w77-94w6.
RUN conda install -y -n base -c conda-forge 'pip=26.2.0' 'anyio=4.14.2' && \
    conda run -n ptca python -m pip install --no-cache-dir --upgrade 'pip==26.2.0' && \
    rm -rf /opt/conda/lib/python3.14/site-packages/pip-26.1*.dist-info \
           /opt/conda/lib/python3.14/site-packages/anyio-4.12*.dist-info \
           /opt/conda/envs/ptca/lib/python3.11/site-packages/pip-26.1*.dist-info && \
    conda clean -ay

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
        -name 'msgpack-1.1.2-py*.egg-info' \
    \) -prune -exec rm -rf {} +; \
    find $roots -type f \( \
        -path '*/conda-meta/setuptools-70.3.0-*.json' -o \
        -path '*/conda-meta/msgpack-python-1.1.2-*.json' -o \
        -path '*/conda-meta/msgpack-1.1.2-*.json' \
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
        -path '*/conda-meta/setuptools-70.3.0-*.json' -o \
        -path '*/conda-meta/msgpack-python-1.1.2-*.json' -o \
        -path '*/conda-meta/msgpack-1.1.2-*.json' \
    \) -print -quit)"; \
    [ -z "$stale" ] || { echo "stale setuptools/msgpack scanner metadata remains: $stale" >&2; exit 1; }

RUN conda clean -a -y && rm -rf /opt/conda/pkgs/ /opt/miniconda/pkgs/

# Reassert newly patched transitive dependencies after all package installs.
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade 'urllib3>=2.8.0' && \
    /opt/conda/envs/ptca/bin/python -m pip install --no-cache-dir --upgrade 'oauthlib>=4.0.0'

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
