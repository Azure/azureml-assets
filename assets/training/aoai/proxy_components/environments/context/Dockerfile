FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade "pip>=25.3"

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install mlflow==3.1.0 "starlette>=0.49.1"

# Security vulnerability fixes
RUN pip install --upgrade PyNaCl==1.6.2