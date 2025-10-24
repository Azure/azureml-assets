#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch271:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install mlflow==3.1.0
RUN pip install transformers==4.53.0

# Security vulnerability fixes
# Fix GHSA-887c-mr87-cxwp: Upgrade torch to 2.8.0
RUN pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 --no-cache-dir
RUN /opt/conda/envs/ptca/bin/pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 || true
# Fix GHSA-4xh5-x5gv-qwph: Upgrade pip to latest secure version
RUN pip install --upgrade pip
RUN /opt/conda/bin/pip install --upgrade pip || true
RUN /opt/conda/envs/ptca/bin/pip install --upgrade pip || true

# Upgrade requests in the system Python (3.13) for fixing vulnerability
RUN /opt/conda/bin/python3.13 -m pip install --upgrade requests urllib3 || true

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
