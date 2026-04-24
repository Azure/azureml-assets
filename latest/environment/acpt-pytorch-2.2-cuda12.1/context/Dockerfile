FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202601.1

# Upgrade base conda packages
RUN conda install -n base -c conda-forge pip=26.0 -y
# These are from base image adding till we get new base image tag
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
# aiohttp 3.12.14 (8 CVEs) and urllib3 2.5.0 (3 CVEs) are also in base env from base image
RUN conda run -n base python -m pip install --upgrade 'wheel>=0.46.2' 'cryptography>=46.0.7' 'setuptools>=82.0.1' 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'urllib3>=2.6.3' 'requests>=2.33.0'


# Upgrade ptca environment conda packages
RUN conda install -c conda-forge pip=26.0 -y && \
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

# Fix Vulnerability 
# TODO: latest version of azureml-mlflow ~ 1.62.0 depends on cryptography<46.0.0
RUN pip install --upgrade --no-cache-dir 'cryptography>=46.0.7'
# CVE-2026-23949 (jaraco.context) & CVE-2026-24049 (wheel) - upgrade setuptools in ptca and base envs
# setuptools vendors jaraco.context internally; --force-reinstall --no-deps ensures vendored copies are replaced
RUN pip install --force-reinstall --no-deps 'setuptools>=82.0.1'
RUN conda run -n base pip install --force-reinstall --no-deps 'setuptools>=82.0.1'