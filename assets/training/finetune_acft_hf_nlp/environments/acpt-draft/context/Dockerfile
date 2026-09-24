FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root
ENV PIP_PROGRESS_BAR=off

# Apply OS security updates not yet covered by the base image.
# libpam and glibc packages are inherited from the base image; fixes USN-8601-1 and USN-8611-1.
# libsystemd0, libudev1 and udev are inherited from the base image; fixes USN-8626-1.
# OpenSSL, curl, and Perl packages are inherited from the base image; fixes USN-8678-1,
# USN-8670-2, and USN-8675-1.
# p11-kit, attr, libssh, diffutils, coreutils, Libgcrypt, libevent, OpenSSH, glibc, .NET,
# Python, and SQLite packages are inherited from the base image; fixes USN-8687-1, USN-8691-1,
# USN-8699-1, USN-8692-1, USN-8697-1, USN-8711-1, USN-8710-1, USN-8721-1, USN-8737-1,
# USN-8740-1, USN-8744-1, and USN-8775-1.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpam-runtime=1.4.0-11ubuntu2.7 \
        libpam0g=1.4.0-11ubuntu2.7 \
        libpam-modules=1.4.0-11ubuntu2.7 \
        libpam-modules-bin=1.4.0-11ubuntu2.7 \
        libc-bin=2.35-0ubuntu3.15 \
        libc-dev-bin=2.35-0ubuntu3.15 \
        libc6=2.35-0ubuntu3.15 \
        libc6-dev=2.35-0ubuntu3.15 \
        locales=2.35-0ubuntu3.15 \
        libgnutls30 rsync \
        dotnet-host-8.0=8.0.31-0ubuntu1~22.04.1 \
        dotnet-hostfxr-8.0=8.0.31-0ubuntu1~22.04.1 \
        dotnet-runtime-8.0=8.0.31-0ubuntu1~22.04.1 \
        libgcrypt20=1.9.4-3ubuntu3.3 \
        liblzma5 xz-utils \
        wget \
        tar=1.34+dfsg-1ubuntu0.1.22.04.6 \
        libkrb5support0=1.19.2-2ubuntu0.8 \
        libgssapi-krb5-2=1.19.2-2ubuntu0.8 \
        libk5crypto3=1.19.2-2ubuntu0.8 \
        libkrb5-3=1.19.2-2ubuntu0.8 \
        libsqlite3-0=3.37.2-2ubuntu0.8 \
        libasound2=1.2.6.1-1ubuntu1.2 \
        libasound2-data=1.2.6.1-1ubuntu1.2 \
        openssh-client=1:8.9p1-3ubuntu0.17 \
        libssl-dev=3.0.2-0ubuntu1.29 \
        libssl3=3.0.2-0ubuntu1.29 \
        openssl=3.0.2-0ubuntu1.29 \
        libcurl4=7.81.0-1ubuntu1.27 \
        curl=7.81.0-1ubuntu1.27 \
        libcurl3-gnutls=7.81.0-1ubuntu1.27 \
        libperl5.34=5.34.0-3ubuntu1.9 \
        perl=5.34.0-3ubuntu1.9 \
        perl-modules-5.34=5.34.0-3ubuntu1.9 \
        perl-base=5.34.0-3ubuntu1.9 \
        libp11-kit0=0.24.0-6ubuntu0.1 \
        libattr1=1:2.5.1-1ubuntu0.1 \
        libssh-4=0.9.6-2ubuntu0.22.04.8 \
        diffutils=1:3.8-0ubuntu2.1 \
        coreutils=8.32-4.1ubuntu1.4 \
        libevent-core-2.1-7=2.1.12-stable-1ubuntu0.1 \
        python3.10=3.10.12-1~22.04.18 \
        python3.10-minimal=3.10.12-1~22.04.18 \
        libpython3.10-minimal=3.10.12-1~22.04.18 \
        libpython3.10-stdlib=3.10.12-1~22.04.18 \
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
# GHSA-69w3-r845-3855 (CVE-2026-1839): arbitrary code execution in Trainer class; patched
# only in transformers>=5.0.0rc3. GHSA-xrqw-3rrv-vx5w: fixed in >=5.10.0. Upgrading to 5.10.0.
# GHSA-86qp-5c8j-p5mr: starlette vulnerability; patched in >=1.0.1 (ptca env).
# azureml-metrics[all] is a transitive dependency of azureml-evaluate-mlflow;
# pinned to the matching latest release to avoid obsolete releases with stale dependencies.
# aiohttp is a transitive dependency from the base Python 3.14 env and AzureML/MLflow stack;
# pinned to fix GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, and GHSA-mfx4-hv73-q22v.
# anyio is a transitive dependency from the base Python 3.14 env (via the httpx/starlette
# and MLflow stack); pinned to fix GHSA-5p39-cfhj-2xmp and GHSA-82r6-8w77-94w6.
# cryptography is a transitive dependency in the ptca env from the AzureML/MLflow stack;
# pinned to fix GHSA-g6cj-pr64-35w5.
# sqlparse is a transitive dependency of MLflow in the ptca env; pinned to fix
# GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, and GHSA-pwgv-4x5q-6m9f.
# Override vulnerable transitive dependencies in the base (Python 3.14) and ptca
# (Python 3.10) environments because parent ranges permit older versions.
# Remove obsolete conda and vendored dependency metadata after upgrading.
RUN pip install -r requirements.txt --no-cache-dir \
 && pip install --no-cache-dir --force-reinstall "sglang>=0.5.10" \
 && pip install --no-cache-dir azureml-acft-common-components=={{latest-pypi-version}} \
 && pip install --no-cache-dir numpy==2.2.5 \
 && pip install --no-cache-dir azureml-evaluate-mlflow=={{latest-pypi-version}} azureml-metrics[all]=={{latest-pypi-version}} \
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
        'python-dotenv>=1.2.2' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'pip>=26.1' \
        'PyJWT>=2.13.0' 'py-rattler>=0.24.0' 'cryptography>=50.0.0' 'pydantic-settings>=2.14.2' 'msgpack>=1.2.1' \
        'setuptools>=83.0.0' 'anyio>=4.14.2' \
 && pip install --no-cache-dir --upgrade 'pip>=26.1' 'PyJWT>=2.13.0' 'msgpack>=1.2.1' 'setuptools>=83.0.0' 'sqlparse>=0.6.0' \
 && find /opt/conda /opt/conda/envs/ptca -path '*/conda-meta/pip-*.json' -delete \
 && find /opt/conda /opt/conda/envs/ptca \( -path '*/conda-meta/setuptools-*.json' -o -path '*/conda-meta/msgpack-python-*.json' \) -delete \
 && find /opt/conda /opt/conda/envs/ptca -path '*/site-packages/pip/_vendor/bom.cdx.json' -delete \
 && rm -rf ~/.cache/pip /root/.cache/pip /tmp/pip-*
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
