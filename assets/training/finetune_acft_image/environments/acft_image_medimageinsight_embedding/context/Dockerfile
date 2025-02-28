# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch222:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update

# Install unzip
RUN apt-get -y install unzip

# Install required packages from pypi
COPY requirements.txt .
COPY MainzVision-1.0.0-py3-none-any.whl .
COPY MainzTrain-0.2.0-py3-none-any.whl .
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install MainzVision-1.0.0-py3-none-any.whl
RUN pip install MainzTrain-0.2.0-py3-none-any.whl
