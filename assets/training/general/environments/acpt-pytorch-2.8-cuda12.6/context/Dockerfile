FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN apt-get update && \
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

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2

# Fix OS-level security vulnerabilities:
# curl (USN-8062-1), libxml2 (USN-7974-1), OpenSSL (USN-7980-1),
# glibc (USN-8005-1), Python (USN-8018-1), GLib (USN-8017-1),
# Expat (USN-8022-1), .NET (USN-8025-1), libpng (USN-8039-1),
# nginx (USN-8038-1), GnuTLS (USN-8043-1), alsa-lib (USN-8044-1)
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix pip security vulnerabilities
RUN pip install --upgrade --no-cache-dir 'pip>=26.0' 'wheel>=0.46.2' 'cryptography>=46.0.5' 'jaraco.context>=6.1.0' 'pillow>=12.1.1' 'protobuf>=6.33.5'
RUN conda run -n base python -m pip install --upgrade --no-cache-dir 'pip>=26.0' 'wheel>=0.46.2' 'cryptography>=46.0.5' 'jaraco.context>=6.1.0' 'pillow>=12.1.1' 'protobuf>=6.33.5'
