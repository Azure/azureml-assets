#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131@sha256:e46b1d4012cca016b7228cc5f7d2b5b37f898426cc6dc6822a1e93ad8575cb74

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install "azureml-evaluate-mlflow==0.0.14.post1"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
