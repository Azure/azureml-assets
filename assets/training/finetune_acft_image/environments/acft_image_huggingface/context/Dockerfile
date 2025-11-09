# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get install -y expat libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl

# Install required packages from pypi
COPY requirements.txt .
RUN pip install --upgrade "pip>=25.3" "starlette>=0.49.1"
RUN pip install -r requirements.txt --no-cache-dir

# Vulnerability Fix
RUN pip install gunicorn==23.0.0 \
    certifi==2024.07.04

# Upgrade requests, pip, starlette in the system Python (3.13) for fixing vulnerability
RUN /opt/conda/bin/python3.13 -m pip install --upgrade pip requests urllib3 starlette || true
RUN /opt/conda/envs/ptca/bin/pip install --upgrade pip starlette || true
