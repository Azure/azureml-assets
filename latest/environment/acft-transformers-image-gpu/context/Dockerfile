# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch210:biweekly.202311.3.v1

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Vulnerability fixes
RUN pip install pydash==6.0.0
RUN pip install urllib3==2.0.7