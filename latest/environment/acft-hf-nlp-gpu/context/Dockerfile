#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch271:biweekly.202507.1

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
# The below 2 files are required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
COPY finetune_run.py /azureml/finetune/run.py

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install mlflow==3.1.0
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN MAX_JOBS=4 pip install flash-attn==2.8.1 --no-build-isolation
RUN pip install nltk==3.9.1 # Pinning to fix the unsafe deserialization vulnerability

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
