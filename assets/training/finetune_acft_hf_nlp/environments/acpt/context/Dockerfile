#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch271:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get update && apt-get -y upgrade

# --- Security Vulnerability Fixes ---
# Fix dpkg vulnerability (USN-7768-1)
RUN apt-get install -y --only-upgrade \
    dpkg-dev=1.21.1ubuntu2.6 \
    dpkg=1.21.1ubuntu2.6 \
    libdpkg-perl=1.21.1ubuntu2.6

# Fix GNU C Library vulnerability (USN-7760-1)
RUN apt-get install -y --only-upgrade \
    libc-bin=2.35-0ubuntu3.11 \
    libc6-dev=2.35-0ubuntu3.11 \
    libc-dev-bin=2.35-0ubuntu3.11 \
    locales=2.35-0ubuntu3.11 \
    libc6=2.35-0ubuntu3.11

# Fix libxml2 vulnerability (USN-7743-1)
RUN apt-get install -y --only-upgrade \
    libxml2=2.9.13+dfsg-1ubuntu0.9

RUN pip install --upgrade pip

COPY requirements.txt .
# The below 2 files are required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
COPY finetune_run.py /azureml/finetune/run.py

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install mlflow==3.1.0
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN MAX_JOBS=4 pip install flash-attn==2.8.1 --no-build-isolation
RUN pip install nltk==3.9.1 # Pinning to fix the unsafe deserialization vulnerability

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
