# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202602.1

USER root
RUN apt-get -y update && apt-get install -y expat libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3>=2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2
