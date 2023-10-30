# PTCA image
FROM mcr.microsoft.com/aifx/acpt/vision/stable-ubuntu2004-cu117-py38-torch1131:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

RUN apt-get -y update

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# Vulnerability fixes
RUN pip install pydash==6.0.0
RUN pip install urllib3==2.0.7