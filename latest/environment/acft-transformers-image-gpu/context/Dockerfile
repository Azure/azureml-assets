# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202602.2

USER root
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y expat libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel, protobuf vulnerabilities in ptca env
# protobuf parent is mlflow==3.5.0 (already latest) which accepts protobuf>=3.12.0, cannot force 6.33.5 via parent
RUN pip install --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3>=2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 pip==26.0 wheel==0.46.2 setuptools==82.0.0
