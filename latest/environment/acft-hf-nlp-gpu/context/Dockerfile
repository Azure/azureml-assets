#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch200:biweekly.202308.2

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install "azureml-evaluate-mlflow==0.0.25"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
