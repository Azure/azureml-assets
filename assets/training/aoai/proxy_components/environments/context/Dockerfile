FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Security: upgrade OpenSSH to patched USN-8222-1 versions, then update the rest of the OS packages.
RUN apt-get update \
    && apt-get install -y --only-upgrade \
        openssh-client=1:9.6p1-3ubuntu13.16 \
        openssh-server=1:9.6p1-3ubuntu13.16 \
        openssh-sftp-server=1:9.6p1-3ubuntu13.16 \
    && apt-get -y upgrade \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
