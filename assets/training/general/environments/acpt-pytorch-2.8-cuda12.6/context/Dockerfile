FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN apt-get update && \
    apt-get upgrade -y && \
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
    apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix ptca conda env vulnerabilities (python 3.10, active).
# pip>=26.1.1: GHSA-jp4c-xjxw-mgf9. pip is a root tool with no parent package,
# and VCM reads the ptca conda-meta record, so conda must update it. The current
# base plus requirements dry-run already keep urllib3 at 2.7.0.
# setuptools>=82.0.1: GHSA-58pv-8j8x-9vj2, GHSA-8rrh-rw8j-w5fx.
# urllib3>=2.7.0: GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j.
# idna>=3.15: GHSA-65pc-fj4g-8rjx.
# aiohttp>=3.14.0: GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8.
RUN conda install -n ptca -c conda-forge -y 'pip>=26.1.1' 'setuptools>=82.0.1' && \
    conda run -n ptca python -m pip install --upgrade \
        'urllib3>=2.7.0' 'idna>=3.15' 'aiohttp>=3.14.0'

# Fix base conda env vulnerabilities (python 3.13). Parent packages still expose
# loose floors as of 2026-05-25, so these direct floors are required:
# setuptools>=82.0.1: conda 26.3.2 requires setuptools>=60.0.0 only.
# python-dotenv>=1.2.2: pydantic-settings 2.14.1 requires >=0.21.0.
# urllib3>=2.7.0: requests 2.34.2 requires <3,>=1.26.
# idna>=3.15: requests 2.34.2 requires <4,>=2.5.
# click>=8.3.3: typer 0.25.1 requires >=8.2.1; mlflow-skinny 3.12.0 requires
#   <9,>=7.0.
# aiohttp>=3.14.0: GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8.
RUN conda run -n base python -m pip install --upgrade \
    'setuptools>=82.0.1' 'python-dotenv>=1.2.2' \
    'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'aiohttp>=3.14.0'

# pip>=26.1.1 in base env: GHSA-jp4c-xjxw-mgf9. pip is a root tool with no
# parent package. Pin conda to the base version because an unconstrained
# conda-forge solve upgrades conda and installs py-rattler 0.23.2, which brings
# the Rust vulnerabilities reported in VCM under rattler/rattler.abi3.so.
RUN conda install -n base -c conda-forge -y 'pip>=26.1.1' 'conda=26.3.2'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
