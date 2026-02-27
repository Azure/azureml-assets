# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade

# Install unzip
RUN apt-get -y install unzip libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel and transitive deps to fix vulnerabilities
RUN pip install --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5 pyasn1>=0.6.2

# Upgrade requests, urllib3, aiohttp in the system Python (3.13) for fixing vulnerability
RUN /opt/conda/bin/python3.13 -m pip install --upgrade requests==2.32.4 urllib3==2.6.3 aiohttp==3.13.3 pip==26.0 wheel==0.46.2 setuptools==82.0.0
