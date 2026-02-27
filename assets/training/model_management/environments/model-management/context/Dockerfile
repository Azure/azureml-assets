FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y wget nscd python3-idna git libxml2 curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Go (if required)
RUN wget https://go.dev/dl/go1.24.13.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.24.13.linux-amd64.tar.gz && \
    rm go1.24.13.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:$PATH"

COPY requirements.txt .

# Clean conda cache
RUN conda clean --all -y

# Ensure we operate inside the correct conda env (ptca)
SHELL ["conda", "run", "-n", "ptca", "/bin/bash", "-c"]

# 🔐 Fix Vulnerabilities (pip / pillow / cryptography)
RUN python -m pip install --upgrade --force-reinstall \
    "pip>=26.0" \
    "wheel>=0.46.2" \
    "setuptools>=75.0.0" \
    cryptography==46.0.5 \
    Pillow==12.1.1 \
    --no-cache-dir

# Install project requirements
RUN python -m pip install --no-cache-dir -r requirements.txt

# 🧹 Remove ALL Azure ML SDK v1 packages (even if base image contains them)
RUN python -m pip uninstall -y \
    azureml-core \
    azureml-sdk \
    azureml-defaults \
    azureml-telemetry \
    azureml-dataset-runtime \
    azureml-train-core \
    azureml-automl-core \
    azureml-automl-runtime \
    azureml-automl-dnn-vision || true

# Also remove from conda level (base image safety)
RUN conda remove -n ptca azureml-core -y || true

# Final verification (CI-safe)
RUN python -m pip freeze | grep azure || true

RUN rm requirements.txt
