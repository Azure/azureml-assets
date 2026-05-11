FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260507.v1

# Security: upgrade all OS packages to patch USN vulnerabilities (incl. USN-8222-1 openssh 1:9.6p1-3ubuntu13.16)
# Reinstall openssh-* explicitly so the upgrade is forced even if the base image pins a held version.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install --reinstall -y openssh-client openssh-server openssh-sftp-server && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Security: upgrade pip to fix CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9). Base image
# openmpi5.0-ubuntu24.04:20260409.v4 ships pip 26.0.1; need >=26.1. pip is its own
# parent (no upstream package to bump), so direct upgrade is the only fix.
RUN pip install --no-cache-dir --upgrade 'pip>=26.1'

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
