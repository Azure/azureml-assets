FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202601.1

# Upgrade pip in base conda env (CVE-2026-1703 requires pip>=26.0; base image ships pip 25.3)
RUN conda install -n base -c conda-forge 'pip>=26.0' -y
# Base image packages requiring security upgrades (base image biweekly.202601.1 not yet patched):
# PyJWT 2.10.1 (CVE-2026-32597), aiohttp 3.12.14, urllib3 2.5.0, python-dotenv 1.1.0 (GHSA-mf9w-mj56-hr94),
# cryptography 44.0.1, setuptools 80.9.0, wheel 0.45.1, requests 2.32.4
# No parent pip packages in the base env pull these at fixed versions; direct overrides required.
RUN conda run -n base python -m pip install --upgrade 'wheel>=0.46.2' 'cryptography>=46.0.7' 'setuptools>=82.0.1' 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'urllib3>=2.6.3' 'requests>=2.33.0' 'python-dotenv>=1.2.2'

# Upgrade pip in ptca conda env (CVE-2026-1703 requires pip>=26.0; base image ships pip 25.3)
RUN conda install -c conda-forge 'pip>=26.0' -y && \
    conda clean -a -y

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

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client

# Fix Vulnerability: cryptography (azureml-mlflow 1.62.0 allows cryptography<47.0.0)
RUN pip install --upgrade --no-cache-dir 'cryptography>=46.0.7'
# CVE-2026-23949 (jaraco.context) & CVE-2026-24049 (wheel) - upgrade setuptools in ptca and base envs
# setuptools vendors jaraco.context internally; --force-reinstall --no-deps ensures vendored copies are replaced
RUN pip install --force-reinstall --no-deps 'setuptools>=82.0.1'
RUN conda run -n base pip install --force-reinstall --no-deps 'setuptools>=82.0.1'