FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Upgrade OS packages to pull current Ubuntu security patches.
# libgcrypt20 is inherited from the openmpi base image; pin to the
# USN-8711-1 patched version for CVE-2024-2236.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade \
        libgcrypt20=1.10.3-2ubuntu0.2 \
        libbrotli1 \
        nginx \
        nginx-common \
        nginx-light && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# pinned transitive dep of azure-identity via msal; azureml-mlflow caps cryptography below the patched version, fixes GHSA-g6cj-pr64-35w5.
RUN pip install -r requirements.txt --no-cache-dir \
    && pip install "cryptography>=50.0.0" --no-cache-dir

# Reassert newly patched transitive dependencies after all package installs.
RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade \
    'PyJWT>=2.15.0' 'oauthlib>=4.0.0' 'pyasn1>=0.6.5' 'setuptools>=84.1.0'

# Reassert November 2026 OS-package floors for installed packages only.
RUN set -eux; \
    apt-get update; \
    security_packages="nginx nginx-common nginx-light libbrotli1 libpcre2-8-0"; \
    installed_packages=""; \
    for package in $security_packages; do \
        if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then \
            installed_packages="$installed_packages $package"; \
        fi; \
    done; \
    if [ -n "$installed_packages" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade $installed_packages; \
    fi; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*
