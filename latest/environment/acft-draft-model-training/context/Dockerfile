FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202609.2
USER root
ENV PIP_PROGRESS_BAR=off

# Apply current OS security updates and ensure required packages are installed.
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
        libpam-runtime \
        libpam0g \
        libpam-modules \
        libpam-modules-bin \
        libc-bin \
        libc-dev-bin \
        libc6 \
        libc6-dev \
        locales \
        libgnutls30 rsync \
        dotnet-host-8.0 dotnet-hostfxr-8.0 dotnet-runtime-8.0 \
        libgcrypt20 \
        liblzma5 xz-utils \
        wget \
        tar \
        libkrb5support0 \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libsqlite3-0 \
        libasound2 \
        libasound2-data \
        openssh-client \
        libssl-dev \
        libssl3 \
        openssl \
        libcurl4 \
        curl \
        libcurl3-gnutls \
        libperl5.34 \
        perl \
        perl-modules-5.34 \
        perl-base \
 && apt-get install -y --only-upgrade libsystemd0 libudev1 udev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# GHSA-jx93-g359-86wm, GHSA-hvwj-8w5g-28rg: sglang vulnerabilities; patched in >=0.5.10
# specforge 0.1.0 pins sglang==0.5.5; override needed after specforge install.
# Upgrade wandb and remove its bundled wandb-core Go binary; the binary carries
# vulnerable Go modules (stdlib/grpc/x/text) and wandb falls back to its Python backend.
# torch is a transitive dep of SpecForge/sglang; this image's pip layers override the
# base torch 2.8.0+cu126 and must stay above GHSA-rrmf-rvhw-rf47's patched floor.
# GHSA-7rgv-gqhr-fxg3: xgrammar stack exhaustion DoS; patched in 0.1.32
# GHSA-69w3-r845-3855 (CVE-2026-1839): arbitrary code execution in Trainer class;
# GHSA-xrqw-3rrv-vx5w requires transformers>=5.10.0.
# GHSA-86qp-5c8j-p5mr: starlette vulnerability; patched in >=1.0.1 (ptca env).
# azureml-metrics[all] is a transitive dependency of azureml-evaluate-mlflow;
# pinned to the matching latest release to avoid obsolete releases with stale dependencies.
# aiohttp is a transitive dependency from the base Python 3.14 env and AzureML/MLflow stack;
# pinned to fix GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, and GHSA-mfx4-hv73-q22v.
# cryptography is a transitive dependency in the ptca env from the AzureML/MLflow stack;
# pinned to fix GHSA-g6cj-pr64-35w5.
# sqlparse is a transitive dependency of MLflow in the ptca env; pinned to fix
# GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, and GHSA-pwgv-4x5q-6m9f.
# Override vulnerable transitive dependencies in the base (Python 3.14) and ptca
# (Python 3.10) environments because parent ranges permit older versions.
# Remove only obsolete pip/setuptools records and stale dependency SBOM metadata after upgrading.
# virtualenv's PEP 770 SBOM advertises its Python 3.9 seed wheels, including pip 26.0.1,
# even though this image's functional Python 3.10/3.14 pip installations are upgraded.
RUN pip install -r requirements.txt --no-cache-dir \
 && pip install --no-cache-dir --force-reinstall "sglang>=0.5.10" \
 && pip install --no-cache-dir azureml-acft-common-components==0.0.91 \
 && pip install --no-cache-dir numpy==2.2.5 \
 && pip install --no-cache-dir azureml-evaluate-mlflow==0.0.91 azureml-metrics[all]==0.0.91 \
 && pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0" \
 && pip install --no-cache-dir --upgrade "wandb>=0.28.1" \
 && find /opt/conda/envs/ptca -name 'wandb-core' -path '*/wandb/bin/*' -delete \
 && pip install --no-cache-dir --upgrade "torch>=2.13.0" \
 && pip install --no-cache-dir xgrammar==0.1.32 \
 && pip install --no-cache-dir transformers==5.10.0 \
 && pip install --no-cache-dir 'starlette>=1.0.1' \
 && pip install --no-cache-dir --upgrade 'aiohttp>=3.14.3' 'cryptography>=50.0.0' \
 && conda run -n base python -m pip install --no-cache-dir --upgrade 'aiohttp>=3.14.3' \
 && conda run -n base python -m pip install --no-cache-dir --upgrade \
        'python-dotenv>=1.2.2' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'pip>=26.2.0' 'anyio>=4.14.2' \
        'PyJWT>=2.13.0' 'py-rattler>=0.24.0' 'cryptography>=50.0.0' 'pydantic-settings>=2.14.2' 'msgpack>=1.2.1' \
        'setuptools>=83.0.0' \
 && pip install --no-cache-dir --upgrade 'pip>=26.2.0' 'PyJWT>=2.13.0' 'msgpack>=1.2.1' 'setuptools>=83.0.0' 'sqlparse>=0.6.0' \
 && rm -rf /opt/conda/lib/python3.14/site-packages/pip-26.0.1.dist-info \
           /opt/conda/lib/python3.14/site-packages/setuptools-82.0.1.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0.1.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/setuptools-82.0.1.dist-info \
 && rm -f /opt/conda/conda-meta/pip-26.0.1-*.json \
          /opt/conda/conda-meta/setuptools-82.0.1-*.json \
          /opt/conda/envs/ptca/conda-meta/pip-26.0.1-*.json \
          /opt/conda/envs/ptca/conda-meta/setuptools-82.0.1-*.json \
          /opt/conda/lib/python3.14/site-packages/pip/_vendor/bom.cdx.json \
          /opt/conda/lib/python3.14/site-packages/pip/_vendor/vendor.txt \
          /opt/conda/lib/python3.14/site-packages/virtualenv-*.dist-info/sboms/virtualenv.cdx.json \
          /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/bom.cdx.json \
          /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/vendor.txt \
          /opt/conda/envs/ptca/lib/python3.10/site-packages/virtualenv-*.dist-info/sboms/virtualenv.cdx.json \
 && stale_metadata="$(find /opt/conda/lib/python3.14/site-packages \
                           /opt/conda/envs/ptca/lib/python3.10/site-packages \
                           /opt/conda/conda-meta /opt/conda/envs/ptca/conda-meta \
                           \( -path '*/pip-26.0.1.dist-info' \
                              -o -path '*/setuptools-82.0.1.dist-info' \
                              -o -path '*/conda-meta/pip-26.0.1-*.json' \
                              -o -path '*/conda-meta/setuptools-82.0.1-*.json' \
                              -o -path '*/pip/_vendor/bom.cdx.json' \
                              -o -path '*/pip/_vendor/vendor.txt' \
                              -o -path '*/virtualenv-*.dist-info/sboms/virtualenv.cdx.json' \) -print -quit)" \
 && test -z "$stale_metadata" \
 && rm -rf ~/.cache/pip /root/.cache/pip /tmp/pip-*
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
