#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131:{{latest-image-tag:biweekly\.\d{6}\.\d{1}}}

USER root
RUN apt-get -y update
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install "azureml-evaluate-mlflow==0.0.16"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
