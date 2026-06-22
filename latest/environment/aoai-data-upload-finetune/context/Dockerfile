FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260614.v1

# Pin OpenSSL OS packages inherited from the base image; fixes USN-8414-1 (vuln 6035833).
RUN apt-get update \
    && apt-get install -y --only-upgrade --no-install-recommends \
        libgnutls30t64 \
        openssl=3.0.13-0ubuntu3.11 \
        libssl3t64=3.0.13-0ubuntu3.11 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
