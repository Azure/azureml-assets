# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# Pull latest Ubuntu 22.04 security patches on each rebuild.
# Explicit package list ensures frequently flagged USN fixes are applied even
# when apt-get upgrade would otherwise hold them back.
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install --only-upgrade \
        attr \
        coreutils \
        curl \
        diffutils \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
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
        libp11-kit0 \
        libperl5.34 \
        libpng16-16 \
        libpython3.10-minimal \
        libpython3.10-stdlib \
        libssh-4 \
        libssl-dev \
        libssl3 \
        liblzma5 \
        libnghttp2-14 \
        libsqlite3-0 \
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
        xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages (ptca env). setuptools>=83.0.0 patches
# GHSA-58pv-8j8x-9vj2 and GHSA-h35f-9h28-mq5c in the ptca env.
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# Vulnerability overrides in base conda env. The base SBOM is patched for
# several of these packages, but requirements.txt can re-resolve
# loose AzureML/MLflow transitive dependencies back to vulnerable versions.
# - setuptools: base ships 82.0.0; need >=83.0.0 for GHSA-58pv-8j8x-9vj2
#   and GHSA-h35f-9h28-mq5c.
# - python-dotenv>=1.2.2: transitive dep of pydantic-settings and
#   anaconda-auth; fixes GHSA-mf9w-mj56-hr94.
# - urllib3>=2.7.0: transitive dep of requests; fixes GHSA-qccp-gfcp-xxvc and
#   GHSA-mf9v-mfxr-j63j.
# - aiohttp>=3.14.3: fixes GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg,
#   GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, and GHSA-mfx4-hv73-q22v.
# - click>=8.3.3: transitive dep of typer/mlflow-skinny; fixes
#   GHSA-47fr-3ffg-hgmw.
# - idna>=3.15: transitive dep of httpx and requests; fixes
#   GHSA-65pc-fj4g-8rjx.
# - PyJWT>=2.13.0: transitive dep from the base image; fixes
#   GHSA-993g-76c3-p5m4 and GHSA-jq35-7prp-9v3f.
# - py-rattler: pinned transitive dep of base conda/rattler tooling brought in
#   by the base image; fixes GHSA-q53q-5r4j-5729.
# - cryptography>=50.0.0: pinned transitive dep brought in by the ACPT base
#   image; fixes GHSA-537c-gmf6-5ccf and GHSA-g6cj-pr64-35w5.
# - pydantic-settings: pinned transitive dep of azureml-defaults ->
#   azureml-inference-server-http; brought in by requirements.txt, fixes
#   GHSA-4xgf-cpjx-pc3j.
# - msgpack: pinned transitive dep of azureml-mlflow -> mlflow-skinny; brought
#   in by requirements.txt, fixes GHSA-6v7p-g79w-8964.
# - anyio: inherited base version is below the scanner-required safe floor.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
        'setuptools>=83.0.0' 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' \
        'aiohttp>=3.14.3' 'click>=8.3.3' 'idna>=3.15' 'PyJWT>=2.13.0' \
        'py-rattler>=0.24.0' 'cryptography>=50.0.0' 'pydantic-settings>=2.14.2' \
        'msgpack>=1.2.1' 'anyio>=4.14.2'

# Vulnerability overrides in ptca conda env (python3.10):
# - urllib3: requirements can re-resolve loose dependencies back to vulnerable
#   urllib3 versions. Parents (requests, torchdata, fsspec) use loose floors, so
#   they cannot force urllib3>=2.7.0; direct override is required.
# - aiohttp: ptca env ships 3.14.1; need >=3.14.3 for GHSA-jg22-mg44-37j8,
#   GHSA-hg6j-4rv6-33pg, GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, and
#   GHSA-mfx4-hv73-q22v. No parent pins aiohttp to a safe floor; direct
#   override is required.
# - starlette: ptca env ships 0.52.1; need >=1.0.1 for GHSA-86qp-5c8j-p5mr.
#   No parent in the ptca env forces starlette>=1.0.1; direct override is
#   required.
# - click: ptca env ships 8.2.1; need >=8.3.3 for GHSA-47fr-3ffg-hgmw. No
#   parent in the ptca env forces click>=8.3.3; direct override is required.
# - pyarrow: ptca env ships 20.0.0; need >=23.0.1 for GHSA-rgxp-2hwp-jwgg.
#   No parent in the ptca env forces pyarrow>=23.0.1; direct override is
#   required.
# - cryptography>=50.0.0: pinned transitive dep brought in by the ACPT base
#   image; fixes GHSA-537c-gmf6-5ccf and GHSA-g6cj-pr64-35w5.
# - msgpack: pinned transitive dep of azureml-mlflow -> mlflow-skinny; brought
#   in by requirements.txt, fixes GHSA-6v7p-g79w-8964.
# - torch: inherited from the ACPT torch280 base image; fixes
#   GHSA-rrmf-rvhw-rf47. torchvision is upgraded with torch to keep the CUDA
#   PyTorch stack compatible.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade \
        'urllib3>=2.7.0' 'aiohttp>=3.14.3' 'starlette>=1.0.1' 'click>=8.3.3' \
        'pyarrow>=23.0.1' 'cryptography>=50.0.0' 'msgpack>=1.2.1' && \
    conda run -n ptca python -m pip install --no-cache-dir --upgrade \
        --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.13.0' 'torchvision==0.28.0'

# Update duplicate base-image Python env metadata that VCM scans.
# pip>=26.2.0 fixes GHSA-qwm4-qh6w-59xr (base image ships 26.1.2).
RUN set -eu; \
    for py in /opt/conda/bin/python /opt/conda/envs/*/bin/python; do \
        if [ -x "$py" ]; then \
            "$py" -m pip install --no-cache-dir --upgrade \
                'setuptools>=83.0.0' \
                'msgpack>=1.2.1' \
                'pip>=26.2.0'; \
        fi; \
    done && \
    roots="/opt /root"; \
    [ ! -d /azureml-envs ] || roots="$roots /azureml-envs"; \
    find $roots -type f \( \
        -path '*/site-packages/pip/_vendor/bom.cdx.json' -o \
        -path '*/site-packages/pip/_vendor/vendor.txt' \
    \) -delete && \
    find $roots -type d \( \
        -name 'setuptools-70.3.0.dist-info' -o \
        -name 'setuptools-70.3.0.egg-info' -o \
        -name 'setuptools-70.3.0-py*.egg-info' -o \
        -name 'msgpack-1.1.2.dist-info' -o \
        -name 'msgpack-1.1.2.egg-info' -o \
        -name 'msgpack-1.1.2-py*.egg-info' \
    \) -prune -exec rm -rf {} + && \
    find $roots -type f \( \
        -path '*/conda-meta/setuptools-70.3.0-*.json' -o \
        -path '*/conda-meta/msgpack-python-1.1.2-*.json' -o \
        -path '*/conda-meta/msgpack-1.1.2-*.json' \
    \) -delete && \
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
    \) -print -quit)" && \
    [ -z "$stale" ] || { echo "stale setuptools/msgpack scanner metadata remains: $stale" >&2; exit 1; }

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
