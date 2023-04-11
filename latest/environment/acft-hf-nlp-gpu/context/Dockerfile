#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131@sha256:f77c40936c9835714e07b7ba96cac74f47c366328d91355257bddd58c6c436d5

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

COPY requirements.txt .
RUN pip install "azureml-acft-accelerator==0.0.6"
RUN pip install "azureml-acft-contrib-hf-nlp==0.0.6"
RUN pip install "azureml-evaluate-mlflow==0.0.6"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
