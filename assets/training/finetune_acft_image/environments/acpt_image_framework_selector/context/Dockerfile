# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch212:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

RUN apt-get -y update

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# Vulnerability fixes
RUN pip install pydash==6.0.0
RUN pip install urllib3==2.0.7
RUN pip install pyarrow==14.0.1
RUN pip install aiohttp==3.9.1
# dummy number to change when needing to force rebuild without changing the definition: 1