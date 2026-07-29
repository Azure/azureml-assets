FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Upgrade base-image OS packages that are still vulnerable in openmpi5.0-ubuntu24.04:20260715.v1.
RUN apt-get update \
    && apt-get install -y --only-upgrade --no-install-recommends \
        tar \
        libsqlite3-0 \
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

# cryptography is constrained by azureml-mlflow but pulled transitively by msal/azure-identity; fixes GHSA-537c-gmf6-5ccf.
RUN pip install -r requirements.txt --no-cache-dir \
    && pip install --upgrade --no-deps cryptography==48.0.1 --no-cache-dir
