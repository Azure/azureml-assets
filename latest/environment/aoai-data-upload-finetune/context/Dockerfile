FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260707.v1

# Pin OS packages inherited from the base image; fixes USN-8414-1 and USN-8398-3.
RUN apt-get update \
    && apt-get install -y --only-upgrade --no-install-recommends \
        libgnutls30t64 \
        nginx=1.24.0-2ubuntu7.12 \
        nginx-common=1.24.0-2ubuntu7.12 \
        nginx-light=1.24.0-2ubuntu7.12 \
        openssl=3.0.13-0ubuntu3.11 \
        libssl3t64=3.0.13-0ubuntu3.11 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# cryptography is constrained by azureml-mlflow but pulled transitively by msal/azure-identity; fixes GHSA-537c-gmf6-5ccf.
RUN pip install -r requirements.txt --no-cache-dir \
    && pip install --upgrade --no-deps cryptography==48.0.1 --no-cache-dir
