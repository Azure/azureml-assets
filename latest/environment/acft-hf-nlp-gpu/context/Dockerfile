#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131@sha256:ef83a69c2c35c3773c0f33a1f580026904c94daa7fc7399d300f27edd17878b8

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

COPY requirements.txt .
RUN pip install "azureml-evaluate-mlflow==0.0.9"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
