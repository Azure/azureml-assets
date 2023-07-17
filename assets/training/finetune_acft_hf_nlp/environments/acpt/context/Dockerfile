#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131
# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install "azureml-evaluate-mlflow"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
