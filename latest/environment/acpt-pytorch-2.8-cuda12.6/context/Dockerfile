FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202608.1

ENV PIP_PROGRESS_BAR=off

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir && \
    pip install --upgrade --no-cache-dir \
        'setuptools>=83.0.0' 'msgpack>=1.2.1'

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN printf 'Acquire::Retries "5";\n' > /etc/apt/apt.conf.d/80-retries && \
    sed -i \
        -e 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' \
        -e 's|http://security.ubuntu.com|https://security.ubuntu.com|g' \
        /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        unzip && \
    apt-get install -y --only-upgrade \
        curl \
        gzip \
        libc-bin \
        libc-dev-bin \
        libc6 \
        libc6-dev \
        libasound2 \
        libasound2-data \
        libcurl3-gnutls \
        libcurl4 \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
        libpam0g \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime \
        libnghttp2-14 \
        libpython3.10-minimal \
        libpython3.10-stdlib \
        libsqlite3-0 \
        locales \
        dotnet-host-8.0 \
        dotnet-hostfxr-8.0 \
        dotnet-runtime-8.0 \
        python3.10 \
        python3.10-minimal \
        tar \
        wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# Deepspeed launcher requires passwordless ssh login
RUN apt-get update && \
    apt-get install -y openssh-server openssh-client && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix ptca conda env vulnerabilities (python 3.10, active).
# setuptools>=83.0.0: GHSA-58pv-8j8x-9vj2, GHSA-8rrh-rw8j-w5fx,
# GHSA-h35f-9h28-mq5c.
# msgpack>=1.2.1: transitive dep of azureml-mlflow -> mlflow-skinny; fixes
# GHSA-6v7p-g79w-8964.
# urllib3>=2.7.0: GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j.
# idna>=3.15: GHSA-65pc-fj4g-8rjx.
# aiohttp>=3.14.3: GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8,
#   GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, GHSA-mfx4-hv73-q22v.
# cryptography>=50.0.0: pinned transitive dep of azureml-core; brought in by
#   base image and AzureML SDK dependencies, fixes GHSA-537c-gmf6-5ccf and
#   GHSA-g6cj-pr64-35w5.
# pyarrow>=23.0.1: transitive dep of azureml-dataset-runtime in the ptca env;
#   azureml-dataset-runtime 1.62.0 caps pyarrow<21, so this post-install
#   security override fixes GHSA-rgxp-2hwp-jwgg.
RUN conda run -n ptca python -m pip install --upgrade --no-cache-dir \
        'setuptools>=83.0.0' 'msgpack>=1.2.1' && \
    find /opt/conda/envs/ptca -type d \( \
        -name 'setuptools-70.3.0.dist-info' -o \
        -name 'setuptools-70.3.0-py*.egg-info' -o \
        -name 'setuptools-81.0.0.dist-info' -o \
        -name 'setuptools-81.0.0-py*.egg-info' -o \
        -name 'msgpack-1.1.2.dist-info' \
    \) -prune -exec rm -rf {} + && \
    conda run -n ptca python -m pip install --upgrade --no-cache-dir \
        'urllib3>=2.7.0' 'idna>=3.15' 'aiohttp>=3.14.3' \
        'cryptography>=50.0.0' 'pyarrow>=23.0.1' && \
    conda clean -a -y && rm -rf /opt/conda/pkgs/

# Fix base conda env vulnerabilities (python 3.14). Parent packages still expose
# loose vulnerable floors, so these direct floors are required:
# setuptools>=83.0.0: conda requires setuptools>=60.0.0 only.
# python-dotenv>=1.2.2: pydantic-settings 2.14.1 requires >=0.21.0.
# urllib3>=2.7.0: requests 2.34.2 requires <3,>=1.26.
# idna>=3.15: requests 2.34.2 requires <4,>=2.5.
# click>=8.3.3: typer 0.25.1 requires >=8.2.1; mlflow-skinny 3.12.0 requires
#   <9,>=7.0.
# aiohttp>=3.14.3: GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8,
#   GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, GHSA-mfx4-hv73-q22v.
# cryptography>=50.0.0: pinned transitive dep of the base image, fixes
#   GHSA-537c-gmf6-5ccf and GHSA-g6cj-pr64-35w5.
# msgpack>=1.2.1: transitive dep of azureml-mlflow -> mlflow-skinny; fixes
#   GHSA-6v7p-g79w-8964.
# PyJWT>=2.13.0: transitive dep from the base image; fixes GHSA-jq35-7prp-9v3f
#   and GHSA-993g-76c3-p5m4.
RUN conda run -n base python -m pip install --upgrade \
    'setuptools>=83.0.0' 'msgpack>=1.2.1' 'python-dotenv>=1.2.2' \
    'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'aiohttp>=3.14.3' \
    'cryptography>=50.0.0' 'PyJWT>=2.13.0' && \
    conda clean -a -y && rm -rf /opt/conda/pkgs/

RUN find /opt/conda -type d \( \
        -name 'setuptools-70.3.0.dist-info' -o \
        -name 'setuptools-70.3.0-py*.egg-info' -o \
        -name 'msgpack-1.1.2.dist-info' \
    \) -prune -exec rm -rf {} + && \
    conda run -n base python -m pip install --upgrade --no-cache-dir \
        'setuptools>=83.0.0' 'msgpack>=1.2.1' && \
    conda run -n ptca python -m pip install --upgrade --no-cache-dir \
        'setuptools>=83.0.0' 'msgpack>=1.2.1' && \
    conda clean -a -y && rm -rf /opt/conda/pkgs/

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
