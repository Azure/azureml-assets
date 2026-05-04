# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade

# Install unzip
RUN apt-get -y install unzip libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel, protobuf vulnerabilities in ptca env
# NOTE: mlflow==3.11.1 pins cryptography<47,>=43.0.0; upgrading to 46.0.7 for CVE fix
# protobuf parent is mlflow-tracing==3.11.1 (via mlflow==3.11.1) which accepts protobuf<8,>=3.12.0; pip resolves older; override needed
# pyasn1 is a transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
# parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
# aiohttp: transitive dep of azure-core; parents use loose floors (GHSA-mwh4-6h8g-pg8w etc.)
# onnx: transitive dep of onnxruntime; parent uses onnx>=1.16.0; override needed (GHSA-p433-9wv8-28xj etc.)
# fastmcp: GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767; >=3.2.0 required
# requests: transitive dep of azure-core/mlflow; parents use loose floors (GHSA-gc5v-m9x4-r6x2)
# Mako: transitive dep (mlflow → alembic → Mako); alembic uses unpinned Mako, cannot force via parent
# pytest: standalone test dep from base image; no parent to upgrade
# python-dotenv: transitive dep (mlflow → mlflow-skinny → python-dotenv<2,>=0.19.0);
# mlflow 3.11.1 (latest as of 2026-05-04) allows >=1.2.2 but pip resolves to 1.2.1; override for GHSA-mf9w-mj56-hr94
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5 cryptography==46.0.7 pyasn1==0.6.3 pillow==12.2.0 'fastmcp>=3.2.0' 'requests>=2.33.0' 'aiohttp>=3.13.4' 'onnx>=1.21.0' 'Mako>=1.3.11' 'pytest>=9.0.3' 'python-dotenv>=1.2.2'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y pip>=26.0.1 wheel>=0.46.2 
# vulnerability in base conda env
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.7 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'requests>=2.33.0'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/