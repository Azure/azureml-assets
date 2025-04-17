# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch222:biweekly.202504.1.v1

USER root
RUN apt-get -y update && apt-get install -y expat=2.4.7-1ubuntu0.5

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Vulnerability Fix
RUN pip install gunicorn==22.0.0
RUN pip install certifi==2024.07.04