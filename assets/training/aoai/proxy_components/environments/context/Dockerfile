FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Upgrade OS packages to pull current Ubuntu security patches.
# libgcrypt20 is inherited from the openmpi base image; pin to the
# USN-8711-1 patched version for CVE-2024-2236.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade \
        libgcrypt20=1.10.3-2ubuntu0.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# pinned transitive dep of azure-identity via msal; azureml-mlflow caps cryptography below the patched version, fixes GHSA-g6cj-pr64-35w5.
RUN pip install -r requirements.txt --no-cache-dir \
    && pip install "cryptography>=50.0.0" --no-cache-dir
