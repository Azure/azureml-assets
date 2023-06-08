#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131@sha256:91fdc953a040ad0e0fafb7ca7e3820c5ce60ce3355c9ba81c7c09a86346701c3

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install "azureml-evaluate-mlflow==0.0.14"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
