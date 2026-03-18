# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# USN-8005-1 glibc, USN-8018-1 python, USN-8022-1 expat, USN-8025-1 .NET,
# USN-8039-1 libpng, USN-8043-1 gnutls, USN-8044-1 alsa-lib, USN-8062-1 curl
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y expat libxml2 libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl libpng16-16 libgnutls30 libasound2 curl libcurl4 libcurl3-gnutls

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel, protobuf vulnerabilities in ptca env
# protobuf parent is mlflow (already latest) which accepts protobuf>=3.12.0, cannot force 6.33.5 via parent
# jaraco.context is a transitive dep (via importlib-resources/setuptools); CVE-2026-23949 zip-slip path traversal
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5 cryptography==46.0.5 pillow==12.1.1 'jaraco.context>=6.1.0'

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# vulnerability in base conda env
RUN conda run -n base python -m pip install --no-cache-dir --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.5
