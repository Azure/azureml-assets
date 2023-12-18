#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202312.2.v2

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
# The below 2 files are required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
COPY finetune_run.py /azureml/finetune/run.py

RUN pip install "azureml-evaluate-mlflow==0.0.38"
# pyarrow-hotfix is installed to fix the vulnerability CVE-2023-47248
# This can be removed once datasets is upgraded to >= 2.14.7
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# dummy number to change when needing to force rebuild without changing the definition: 1
