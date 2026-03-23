FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Security: upgrade all OS packages to patch USN vulnerabilities
RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

# Security vulnerability fixes
# pyasn1 override removed: google-auth → pyasn1-modules (>=0.2.1) → pyasn1 (>=0.6.1,<0.7.0)
# pip resolves to latest 0.6.3 (fixed) naturally without explicit override
