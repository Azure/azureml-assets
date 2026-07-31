FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Upgrade base-image OS packages that are still vulnerable in the rendered base image.
RUN apt-get update \
    && apt-get install -y --only-upgrade --no-install-recommends \
        tar \
        libsqlite3-0 \
        rsyslog \
        libpam-runtime \
        libpam0g \
        libpam-modules \
        libpam-modules-bin \
        nginx-common \
        nginx-light \
        nginx \
        wget \
        libkrb5support0 \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
