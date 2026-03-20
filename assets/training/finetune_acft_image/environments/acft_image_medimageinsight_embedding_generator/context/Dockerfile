# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

# Install unzip and upgrade OS packages to fix vulnerabilities
RUN apt-get -y update && apt-get -y upgrade && apt-get -y install unzip libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# protobuf is a transitive dep of mlflow-skinny/onnx; parents use loose floors (>=3.12.0), cannot force 6.33.5
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# setuptools vendors jaraco.context internally; >=82.0.1 bundles the patched version (GHSA-58pv-8j8x-9vj2)

# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -y -n ptca pip>=26.0.1 wheel>=0.46.2

# Fix vulnerabilities from base image transitive deps
# skops 0.11.0 → ≥0.13.0 (CVE-2025-54412, CVE-2025-54413, CVE-2025-54886 — arbitrary code execution)
#   Required-by: mlflow; all mlflow versions (up to 3.11.0rc0) specify skops<1 with no minimum,
#   so no mlflow upgrade can fix this — explicit override is required.
# wheel ≥0.46.2 via pip as fallback in case conda version lags (CVE-2026-24049)
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade 'skops>=0.13.0' 'wheel>=0.46.2'

# Upgrade requests, urllib3, aiohttpin the system Python (3.13) for fixing vulnerability
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 'setuptools>=82.0.1' cryptography==46.0.5 'PyJWT>=2.12.0'
