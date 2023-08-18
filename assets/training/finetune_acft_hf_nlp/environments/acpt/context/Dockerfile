#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch200:biweekly.202308.1.v1

USER root

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install "azureml-evaluate-mlflow=={{latest-pypi-version}}"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
