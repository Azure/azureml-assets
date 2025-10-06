# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get install -y expat

# --- Security Vulnerability Fixes ---
# Fix dpkg vulnerability (USN-7768-1)
RUN apt-get install -y --only-upgrade \
    dpkg-dev=1.21.1ubuntu2.6 \
    dpkg=1.21.1ubuntu2.6 \
    libdpkg-perl=1.21.1ubuntu2.6

# Fix GNU C Library vulnerability (USN-7760-1)
RUN apt-get install -y --only-upgrade \
    libc-bin=2.35-0ubuntu3.11 \
    libc6-dev=2.35-0ubuntu3.11 \
    libc-dev-bin=2.35-0ubuntu3.11 \
    locales=2.35-0ubuntu3.11 \
    libc6=2.35-0ubuntu3.11

# Fix libxml2 vulnerability (USN-7743-1)
RUN apt-get install -y --only-upgrade \
    libxml2=2.9.13+dfsg-1ubuntu0.9

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Vulnerability Fix
RUN pip install gunicorn==23.0.0
RUN pip install certifi==2024.07.04

# Upgrade requests in the system Python (3.13) for fixing vulnerability
RUN /opt/conda/bin/python3.13 -m pip install --upgrade requests urllib3 || true
