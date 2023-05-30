#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131@sha256:c368e4e0f5b7ac9ff002b2a1e3f794af1dc2eef4ff63bc1f39b1072bb7a264a0

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

COPY requirements.txt .
RUN pip install "azureml-evaluate-mlflow==0.0.10"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
