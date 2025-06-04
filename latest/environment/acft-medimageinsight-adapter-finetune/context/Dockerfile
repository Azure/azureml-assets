# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch260:biweekly.202506.1.v1

USER root
RUN apt-get -y update

# Install unzip
RUN apt-get -y install unzip

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
