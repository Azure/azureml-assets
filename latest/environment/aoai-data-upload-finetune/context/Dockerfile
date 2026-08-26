FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260810.v1

# Upgrade OS packages to pull current Ubuntu security patches.
# The current openmpi5.0-ubuntu24.04 base plus apt upgrade contains the required
# Ubuntu security fixes, including the systemd packages from USN-8626-1.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# pinned transitive dep of azure-identity via msal; azureml-mlflow caps cryptography below the patched version, fixes GHSA-g6cj-pr64-35w5.
RUN pip install -r requirements.txt --no-cache-dir \
    && pip install "cryptography>=50.0.0" --no-cache-dir --no-deps
