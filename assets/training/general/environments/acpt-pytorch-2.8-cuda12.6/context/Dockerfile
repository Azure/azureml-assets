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
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client


# Fix security vulnerabilities
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# setuptools vendors jaraco.context internally; >=82.0.1 bundles the patched version (GHSA-58pv-8j8x-9vj2)
RUN pip install --upgrade 'pip>=26.0' 'wheel>=0.46.2' 'cryptography>=46.0.5' 'pillow>=12.1.1' 'protobuf>=6.33.5' 'setuptools>=82.0.1'
# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade 'pip>=26.0' 'wheel>=0.46.2' 'cryptography>=46.0.5' 'pillow>=12.1.1' 'protobuf>=6.33.5' 'setuptools>=82.0.1'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -y -n ptca pip>=26.0.1
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/