FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202509.2

USER root
RUN apt-get -y update && apt-get install -y libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl


COPY requirements.txt .

RUN pip install -r requirements.txt