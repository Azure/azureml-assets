#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202311.2

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install "azureml-evaluate-mlflow==0.0.35"
# pyarrow-hotfix is installed to fix the vulnerability CVE-2023-47248
# This can be removed once datasets is upgraded to >= 2.14.7
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
