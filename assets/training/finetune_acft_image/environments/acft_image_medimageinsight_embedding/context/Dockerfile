# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

# Install unzip and upgrade OS packages to fix vulnerabilities
RUN apt-get -y update && apt-get -y upgrade && apt-get -y install unzip libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel and transitive deps to fix vulnerabilities
# protobuf is a transitive dep of mlflow-skinny/onnx; parents use loose floors (>=3.12.0), cannot force 6.33.5
RUN pip install --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5

# Upgrade packages in the system Python (3.13) for fixing vulnerability
RUN /opt/conda/bin/python3.13 -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 pip==26.0 wheel==0.46.2 setuptools==82.0.0
