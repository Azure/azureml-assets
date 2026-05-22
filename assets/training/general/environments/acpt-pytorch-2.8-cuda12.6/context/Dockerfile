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

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update && \
    apt-get install -y openssh-server openssh-client && \
    apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix security vulnerabilities in ptca conda env (active environment)
# setuptools>=82.0.1: GHSA-58pv-8j8x-9vj2, GHSA-8rrh-rw8j-w5fx; base image has 82.0.0
# pip>=26.1: CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9); base image has pip 26.0.1.
#   pip is a root tool (no parent package) installed directly by conda; only a
#   direct upgrade can fix this.
RUN pip install --upgrade 'setuptools>=82.0.1' 'pip>=26.1'

# Fix security vulnerabilities in base conda env (python 3.13)
# setuptools>=82.0.1: same CVEs as above; base image has 82.0.0
# python-dotenv>=1.2.2: CVE-2026-28684 (GHSA-mf9w-mj56-hr94); base env ships 1.2.1
# pip>=26.1: CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9); base env ships pip 26.0.1.
#   pip is a root tool (no parent package) installed directly by conda; only a
#   direct upgrade can fix this.
RUN conda run -n base python -m pip install --upgrade 'setuptools>=82.0.1' 'python-dotenv>=1.2.2' 'pip>=26.1'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/