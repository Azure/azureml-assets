FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch27x:biweekly.202512.2

WORKDIR /workspace

RUN apt update && \
    apt install -y --no-install-recommends \
        build-essential \
        python3-dev \
        wget \
        fuse3 \
        libfuse3-3 \
        dos2unix \
        ca-certificates \
        rsync \
        strace && \
    wget -q https://github.com/Azure/azure-storage-fuse/releases/download/blobfuse2-2.5.0/blobfuse2-2.5.0-Ubuntu-22.04.x86_64.deb && \
    dpkg -i blobfuse2-2.5.0-Ubuntu-22.04.x86_64.deb || apt install -f -y && \
    rm -f blobfuse2-2.5.0-Ubuntu-22.04.x86_64.deb && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt .
RUN pip install --no-cache-dir numpy==1.26.2 cython==3.0.11 && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt && \
    conda clean -afy && \
    pip cache purge && \
    find /opt/conda -name "*.pyc" -delete

COPY startup.sh .
RUN dos2unix startup.sh && chmod +x startup.sh