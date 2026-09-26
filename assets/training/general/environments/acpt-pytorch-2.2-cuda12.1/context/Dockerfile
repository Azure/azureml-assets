FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

ENV PIP_PROGRESS_BAR=off

# Conda env security floors.
# setuptools>=82.0.1: fixes GHSA-h35f-9h28-mq5c.
# PyJWT>=2.13.0: pinned transitive dep brought in by the base image; fixes GHSA-jq35-7prp-9v3f and GHSA-993g-76c3-p5m4.
# py-rattler>=0.24.0: pinned transitive dep of base conda tooling; brought in by base image, fixes GHSA-q53q-5r4j-5729.
# cryptography>=50.0.0: pinned transitive dep present in the base image; fixes GHSA-537c-gmf6-5ccf.
# pydantic-settings>=2.14.2: pinned transitive dep present in the base image; fixes GHSA-4xgf-cpjx-pc3j.
# msgpack>=1.2.1: pinned transitive dep present in the base image; fixes GHSA-6v7p-g79w-8964.
# aiohttp>=3.14.3: pinned transitive dep present in the base image; fixes GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, and GHSA-mfx4-hv73-q22v.
# anyio>=4.14.2: pinned transitive dep in the base prefix; fixes GHSA-5p39-cfhj-2xmp and GHSA-82r6-8w77-94w6.
RUN conda install -n ptca -c conda-forge -y 'pip>=26.1' 'setuptools>=82.0.1' && \
    conda install -n base -c conda-forge -y 'pip>=26.1' 'setuptools>=82.0.1' && \
    conda run -n base python -m pip install --upgrade --no-cache-dir \
        'setuptools>=82.0.1' \
        'python-dotenv>=1.2.2' \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'click>=8.3.3' \
        'aiohttp>=3.14.3' \
        'anyio>=4.14.2' \
        'PyJWT>=2.13.0' \
        'py-rattler>=0.24.0' \
        'cryptography>=50.0.0' \
        'pydantic-settings>=2.14.2' \
        'msgpack>=1.2.1' && \
    conda run -n ptca python -m pip install --upgrade --no-cache-dir \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'setuptools>=82.0.1' \
        'aiohttp>=3.14.3' \
        'msgpack>=1.2.1' && \
    site_packages="$(find /opt/conda/lib -mindepth 2 -maxdepth 2 -type d \
        -path '/opt/conda/lib/python*/site-packages' -print; \
        if [ -d /opt/conda/envs/ptca/lib/python3.10/site-packages ]; then \
            printf '%s\n' /opt/conda/envs/ptca/lib/python3.10/site-packages; \
        fi)" && \
    for site_packages_dir in $site_packages; do \
        rm -f \
            "$site_packages_dir/pip/_vendor/bom.cdx.json" \
            "$site_packages_dir/pip/_vendor/vendor.txt"; \
        find "$site_packages_dir" -maxdepth 1 -type d \( \
                -name 'setuptools-70.3.0.dist-info' -o \
                -name 'setuptools-70.3.0.egg-info' -o \
                -name 'setuptools-70.3.0-py*.egg-info' -o \
                -name 'msgpack-1.1.2.dist-info' -o \
                -name 'msgpack-1.1.2.egg-info' -o \
                -name 'msgpack-1.1.2-py*.egg-info' \
            \) -prune -exec rm -rf {} +; \
    done && \
    rm -f \
        /opt/conda/conda-meta/setuptools-70.3.0-*.json \
        /opt/conda/conda-meta/msgpack-1.1.2-*.json \
        /opt/conda/conda-meta/msgpack-python-1.1.2-*.json \
        /opt/conda/envs/ptca/conda-meta/setuptools-70.3.0-*.json \
        /opt/conda/envs/ptca/conda-meta/msgpack-1.1.2-*.json \
        /opt/conda/envs/ptca/conda-meta/msgpack-python-1.1.2-*.json && \
    stale_metadata="$(find $site_packages \
        /opt/conda/conda-meta \
        /opt/conda/envs/ptca/conda-meta \( \
            -path '*/pip/_vendor/bom.cdx.json' -o \
            -path '*/pip/_vendor/vendor.txt' -o \
            -name 'setuptools-70.3.0.dist-info' -o \
            -name 'setuptools-70.3.0.egg-info' -o \
            -name 'setuptools-70.3.0-py*.egg-info' -o \
            -name 'setuptools-70.3.0-*.json' -o \
            -name 'msgpack-1.1.2.dist-info' -o \
            -name 'msgpack-1.1.2.egg-info' -o \
            -name 'msgpack-1.1.2-py*.egg-info' -o \
            -name 'msgpack-1.1.2-*.json' -o \
            -name 'msgpack-python-1.1.2-*.json' \
    \) -print -quit)" && test -z "$stale_metadata" && \
    conda clean -a -y && \
    rm -rf /opt/conda/pkgs

# Install pip dependencies (runs in default PTCA conda env)
COPY requirements.txt .
# pyarrow>=23.0.1,<24: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg
# while preserving the parent package's compatibility ceiling.
# numpy<1.27: scipy 1.10.1 compatibility ceiling; prevents transitive security overrides from selecting NumPy 2.x.
# ml-dtypes<0.6: compatible with the NumPy ceiling; 0.6 requires NumPy 2.x.
# cryptography>=50.0.0: pinned transitive dep of azureml-core in requirements.txt; fixes GHSA-537c-gmf6-5ccf.
RUN pip install -r requirements.txt --no-cache-dir && \
    pip install --upgrade --no-cache-dir \
        'numpy<1.27' \
        'ml-dtypes<0.6' \
        'pyarrow>=23.0.1,<24' \
        'cryptography>=50.0.0' \
        'setuptools>=83.0.0' && \
    conda run -n base python -m pip install --upgrade --no-cache-dir 'setuptools>=83.0.0'

# Temporary override for the Torch stack inherited from the ACPT torch280 base;
# resolves the currently non-compliant Torch floors with the highest complete
# matched trio available from the CUDA 12.6 index.
# Remove when the base provides a complete companion stack with torch>=2.13.0.
# datasets>=4.8.4 handles torchvision releases that no longer expose VideoReader.
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade \
        --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0' && \
    /opt/conda/envs/ptca/bin/python -m pip install --upgrade --no-cache-dir \
        'datasets>=4.8.4' 'setuptools>=83.0.0'

# Inference requirements.
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade --no-install-recommends \
        dotnet-host-8.0 \
        dotnet-hostfxr-8.0 \
        dotnet-runtime-8.0 \
        coreutils \
        curl \
        diffutils \
        libasound2 \
        libasound2-data \
        libattr1 \
        libc-bin \
        libc-dev-bin \
        libc6 \
        libc6-dev \
        libcurl3-gnutls \
        libcurl4 \
        libevent-core-2.1-7 \
        libexpat1 \
        libgcrypt20 \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
        libp11-kit0 \
        libperl5.34 \
        libssh-4 \
        libssl-dev \
        libssl3 \
        libsystemd0 \
        libudev1 \
        libxml2 \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime \
        libpam0g \
        libsqlite3-0 \
        locales \
        openssl \
        perl \
        perl-base \
        perl-modules-5.34 \
        tar \
        udev \
        wget && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libnginx-mod-http-echo \
        libnginx-mod-http-geoip2 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        unzip && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade --no-install-recommends \
        libpng16-16 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# Deepspeed launcher requires passwordless ssh; server side not in base image
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server && \
    rm -rf /var/lib/apt/lists/*
