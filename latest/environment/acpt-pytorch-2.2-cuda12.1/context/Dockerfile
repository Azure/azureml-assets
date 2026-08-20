FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202608.1

ENV PIP_PROGRESS_BAR=off

# Conda environment security floors.
# setuptools>=83.0.0: fixes GHSA-h35f-9h28-mq5c.
# PyJWT>=2.13.0: pinned transitive dep brought in by the base image; fixes GHSA-jq35-7prp-9v3f and GHSA-993g-76c3-p5m4.
# py-rattler>=0.24.0: pinned transitive dep of base conda tooling; brought in by base image, fixes GHSA-q53q-5r4j-5729.
# cryptography>=50.0.0: pinned transitive dep present in the base image; fixes GHSA-g6cj-pr64-35w5.
# pydantic-settings>=2.14.2: pinned transitive dep present in the base image; fixes GHSA-4xgf-cpjx-pc3j.
# msgpack>=1.2.1: pinned transitive dep present in the base image; fixes GHSA-6v7p-g79w-8964.
RUN conda install -n ptca -c conda-forge -y 'pip>=26.1' 'setuptools>=83.0.0' && \
    conda install -n base -c conda-forge -y 'pip>=26.1' 'setuptools>=83.0.0' && \
    conda run -n base python -m pip install --upgrade --no-cache-dir \
        'setuptools>=83.0.0' \
        'python-dotenv>=1.2.2' \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'click>=8.3.3' \
        'aiohttp>=3.14.3' \
        'PyJWT>=2.13.0' \
        'py-rattler>=0.24.0' \
        'cryptography>=50.0.0' \
        'pydantic-settings>=2.14.2' \
        'msgpack>=1.2.1' && \
    conda run -n ptca python -m pip install --upgrade --no-cache-dir \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'setuptools>=83.0.0' && \
    conda clean -a -y && \
    rm -rf /opt/conda/pkgs

# Install pip dependencies in the default PTCA conda environment.
COPY requirements.txt .
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
# cryptography>=50.0.0: pinned transitive dep of azureml-core in requirements.txt; fixes GHSA-g6cj-pr64-35w5.
RUN pip install -r requirements.txt --no-cache-dir && \
    pip install --upgrade --no-cache-dir --no-deps \
        'pyarrow>=23.0.1' \
        'cryptography>=50.0.0' \
        'setuptools>=83.0.0' && \
    rm -rf /opt/conda/lib/python3.14/site-packages/setuptools-82.0.0.dist-info && \
    find /opt/conda \( -path '*/pip/_vendor/vendor.txt' -o -path '*/pip/_vendor/bom.cdx.json' \) -delete

# Inference requirements.
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
# The --only-upgrade packages below are inherited unpatched from the ACPT base image, whose
# tag lags the Ubuntu 22.04 security archive. Remove that list once the base image ships the
# fixed versions on its own.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade --no-install-recommends \
        dotnet-host-8.0 \
        dotnet-hostfxr-8.0 \
        dotnet-runtime-8.0 \
        libasound2 \
        libasound2-data \
        libc-bin \
        libc-dev-bin \
        libc6 \
        libc6-dev \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
        locales \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime \
        libpam0g \
        libsqlite3-0 \
        tar \
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
