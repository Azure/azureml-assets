# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202506.3

USER root
RUN apt-get -y update
    
# Install unzip
RUN apt-get -y install unzip

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN rm -rf /opt/conda/lib/python3.13 || true
