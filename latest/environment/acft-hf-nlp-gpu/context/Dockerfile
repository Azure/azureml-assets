#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131@sha256:622c1b282f94ead6d4530f7b7cb08bde67c55f757bcbb0ae13f267f1ce214798

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

COPY requirements.txt .
RUN pip install "azureml-evaluate-mlflow==0.0.6.post1"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
