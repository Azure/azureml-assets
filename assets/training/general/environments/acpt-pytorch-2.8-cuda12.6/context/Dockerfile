FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# Upgrade pip, setuptools, wheel in active conda env
RUN pip install --upgrade "pip>=26.0" "setuptools>=82.0.1" "wheel>=0.46.2"
# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Security fix: GHSA-r6ph-v2qm-q3c2 (cryptography<46.0.5). Must run after requirements.txt
# to prevent transitive deps from downgrading the version.
RUN pip install --upgrade --no-cache-dir 'cryptography>=46.0.5'

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
# Install inference + ssh packages, upgrade all OS packages, and apply targeted security fixes:
# USN-7974-1 (libxml2), USN-7980-1 (openssl/libssl3), USN-8005-1 (libc6),
# USN-8017-1 (libglib2.0-0), USN-8022-1 (libexpat1), USN-8039-1 (libpng16-16),
# USN-8043-1 (libgnutls30), USN-8044-1 (libasound2), USN-8062-1 (curl/libcurl4)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        openssh-client \
        openssh-server \
        psmisc \
        rsyslog \
        runit \
        unzip && \
    apt-get install -y --only-upgrade \
        curl \
        libcurl4 \
        libexpat1 \
        libglib2.0-0 \
        libgnutls30 \
        libpng16-16 \
        libssl3 \
        libxml2 \
        libasound2 \
        libc6 \
        openssl && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# Upgrade pip, setuptools, wheel in base conda env to fix:
# GHSA-6vgw-5pg2-w6jp (pip), GHSA-58pv-8j8x-9vj2 (jaraco.context via setuptools), GHSA-8rrh-rw8j-w5fx (wheel)
RUN conda run -n base python -m pip install --upgrade "pip>=26.0" "setuptools>=82.0.1" "wheel>=0.46.2" "cryptography>=46.0.5"


