FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260715.v1

# Upgrade OS packages inherited vulnerable from the base image until the base image carries these Ubuntu security updates.
RUN apt-get update \
    && apt-get install -y --only-upgrade --no-install-recommends \
        curl \
        gzip \
        libcurl3t64-gnutls \
        libcurl4t64 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# cryptography is constrained by azureml-mlflow but pulled transitively by msal/azure-identity; fixes GHSA-537c-gmf6-5ccf.
RUN pip install -r requirements.txt --no-cache-dir \
    && pip install --upgrade --no-deps cryptography==48.0.1 --no-cache-dir
