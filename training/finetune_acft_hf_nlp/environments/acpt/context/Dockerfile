#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131:{{latest-image-tag}}

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-evaluate-mlflow==0.0.1
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
