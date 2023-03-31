#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131@sha256:7d31cc5607e3352deed945c251b3b81df10d4e1a9f2a145b4faa48bddfd06a29

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

COPY requirements.txt .
RUN pip install azureml-evaluate-mlflow==0.0.1
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
