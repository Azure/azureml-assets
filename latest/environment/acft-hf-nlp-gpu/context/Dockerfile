#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202512.2

USER root

RUN apt-get update && apt-get -y upgrade

COPY requirements.txt .
# The below 2 files are required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
COPY finetune_run.py /azureml/finetune/run.py

RUN pip install -r requirements.txt --no-cache-dir

# pip dependency resolver will break if this is added to requirements.txt
RUN pip install fastmcp>=2.14.0
RUN pip install mlflow==3.1.0
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN MAX_JOBS=$(nproc) pip install --no-cache-dir --upgrade flash-attn==2.8.3 --no-build-isolation
RUN pip install nltk==3.9.1 # Pinning to fix the unsafe deserialization vulnerability

# vulnerabilities
RUN pip install --upgrade --no-cache-dir 'urllib3>=2.6.0'

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
