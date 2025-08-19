#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch271:biweekly.202508.1

USER root

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-acft-common-components==0.0.78
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow==0.0.78
RUN pip install mlflow==3.1.0
RUN pip install transformers==4.52.1

# Upgrade requests in the system Python (3.13) for fixing vulnerability
RUN /opt/conda/bin/python3.13 -m pip install --upgrade requests urllib3 || true

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
