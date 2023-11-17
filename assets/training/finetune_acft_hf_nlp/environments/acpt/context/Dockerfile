#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202310.2

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
# data import code files
COPY data_import.py /azureml/data_import/run.py
# finetune code files
COPY model_selector.py /azureml/finetune/model_selector.py
COPY preprocess.py /azureml/finetune/preprocess.py
COPY finetune.py /azureml/finetune/finetune.py
COPY register_model.py /azureml/finetune/register_model.py
COPY register_presets_model.py /azureml/finetune/register_presets_model.py
COPY zero2.json /azureml/finetune/zero2.json
COPY zero3.json /azureml/finetune/zero3.json

RUN pip install "azureml-evaluate-mlflow==0.0.31"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
