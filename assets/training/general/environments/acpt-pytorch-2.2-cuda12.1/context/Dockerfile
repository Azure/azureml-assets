FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

ENV PIP_PROGRESS_BAR=off

# Conda env security floors.
# setuptools>=82.0.1: fixes GHSA-h35f-9h28-mq5c.
# PyJWT>=2.13.0: pinned transitive dep brought in by the base image; fixes GHSA-jq35-7prp-9v3f and GHSA-993g-76c3-p5m4.
# py-rattler>=0.24.0: pinned transitive dep of base conda tooling; brought in by base image, fixes GHSA-q53q-5r4j-5729.
# cryptography>=48.0.1: pinned transitive dep present in the base image; fixes GHSA-537c-gmf6-5ccf.
# pydantic-settings>=2.14.2: pinned transitive dep present in the base image; fixes GHSA-4xgf-cpjx-pc3j.
# msgpack>=1.2.1: pinned transitive dep present in the base image; fixes GHSA-6v7p-g79w-8964.
# torch==2.13.0+cu126: overrides vulnerable torch from the ACPT torch280 base image; fixes GHSA-rrmf-rvhw-rf47.
RUN conda install -n ptca -c conda-forge -y 'pip>=26.1' 'setuptools>=82.0.1' && \
    conda install -n base -c conda-forge -y 'pip>=26.1' 'setuptools>=82.0.1' && \
    conda run -n base python -m pip install --upgrade --no-cache-dir \
        'setuptools>=82.0.1' \
        'python-dotenv>=1.2.2' \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'click>=8.3.3' \
        'aiohttp>=3.14.0' \
        'PyJWT>=2.13.0' \
        'py-rattler>=0.24.0' \
        'cryptography>=48.0.1' \
        'pydantic-settings>=2.14.2' \
        'msgpack>=1.2.1' && \
    conda run -n ptca python -m pip install --upgrade --no-cache-dir \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'setuptools>=82.0.1' && \
    conda run -n ptca python -m pip install --upgrade --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.13.0+cu126' \
        'torchvision==0.28.0+cu126' && \
    conda run -n ptca python -m pip uninstall -y torchaudio && \
    conda clean -a -y && \
    rm -rf /opt/conda/pkgs

# Install pip dependencies (runs in default PTCA conda env)
COPY requirements.txt .
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
# cryptography>=48.0.1: pinned transitive dep of azureml-core in requirements.txt; fixes GHSA-537c-gmf6-5ccf.
RUN pip install -r requirements.txt --no-cache-dir && \
    pip install --upgrade --no-cache-dir --no-deps \
        'pyarrow>=23.0.1' \
        'cryptography>=48.0.1' \
        'setuptools>=83.0.0' && \
    conda run -n base python -m pip install --upgrade --no-cache-dir --no-deps 'setuptools>=83.0.0' && \
    rm -rf /opt/conda/lib/python3.14/site-packages/setuptools-82.0.0.dist-info && \
    find /opt/conda \( -path '*/pip/_vendor/vendor.txt' -o -path '*/pip/_vendor/bom.cdx.json' \) -delete

# Inference requirements.
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade --no-install-recommends \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
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
