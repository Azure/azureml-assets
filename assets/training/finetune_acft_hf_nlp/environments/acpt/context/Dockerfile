#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202310.2

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
COPY data_import.py /azureml/data_import/run.py

RUN pip install "azureml-evaluate-mlflow==0.0.31"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
