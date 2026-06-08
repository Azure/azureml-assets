FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

RUN apt-get update \
    && apt-get install -y --only-upgrade --no-install-recommends \
        libgnutls30t64 \
        libgcrypt20 \
        liblzma5 \
        xz-utils \
        nginx \
        nginx-common \
        nginx-light \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
