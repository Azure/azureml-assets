#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131@sha256:9591af8913c031eeefb1458a4b039120cb8a065235cdaf789c2bd35ee12c34a4

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install "azureml-evaluate-mlflow==0.0.16"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
