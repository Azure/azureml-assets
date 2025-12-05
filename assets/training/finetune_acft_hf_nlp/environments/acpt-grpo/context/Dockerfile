#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get install -y libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
