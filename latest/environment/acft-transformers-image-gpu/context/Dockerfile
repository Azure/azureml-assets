# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch212:biweekly.202401.3

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Vulnerability fixes
RUN pip install pydash==6.0.0
RUN pip install urllib3==2.0.7
RUN pip install pyarrow==14.0.1
RUN pip install aiohttp==3.9.1
# dummy number to change when needing to force rebuild without changing the definition: 1