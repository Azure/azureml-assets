# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202311.2

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir