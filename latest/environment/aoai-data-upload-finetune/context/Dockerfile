FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260514.v1

# Security: upgrade all OS packages so the image picks up the latest USN-patched
# versions from Ubuntu noble-updates (e.g. USN-8222-1 openssh, USN-8226-1 kmod,
# USN-8227-1 curl, USN-8229-1 sed, USN-8233-1 nghttp2). The base image is the
# parent for these packages; we cannot bump it directly, so an in-image upgrade
# is the standard remediation.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
