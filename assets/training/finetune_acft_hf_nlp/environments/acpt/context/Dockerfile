#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch201
USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install "azureml-evaluate-mlflow==0.0.30"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
