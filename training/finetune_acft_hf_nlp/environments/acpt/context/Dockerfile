#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131:{{latest-image-tag}}

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

COPY requirements.txt .
RUN pip install --upgrade "azureml-acft-accelerator=={{latest-pypi-version}}"
RUN pip install --upgrade "azureml-acft-contrib-hf-nlp=={{latest-pypi-version}}"
RUN pip install --upgrade "azureml-evaluate-mlflow=={{latest-pypi-version}}"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
