FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

RUN apt-get update && apt-get -y upgrade

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
