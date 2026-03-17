# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade

# Install unzip
RUN apt-get -y install unzip libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel, protobuf vulnerabilities in ptca env
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# protobuf parent is mlflow==3.5.0 (already latest) which accepts protobuf>=3.12.0, cannot force 6.33.5 via parent
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5 cryptography==46.0.5 pyasn1==0.6.2 pillow==12.1.1 fastmcp>=2.14.0
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y pip>=26.0.1 wheel>=0.46.2 
# vulnerability in base conda env
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.5

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/