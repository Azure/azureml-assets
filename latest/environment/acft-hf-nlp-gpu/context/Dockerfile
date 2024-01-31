#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202401.3

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
# The below 2 files are required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
COPY finetune_run.py /azureml/finetune/run.py

RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
RUN MAX_JOBS=4 pip install flash-attn==2.3.3 --no-build-isolation

# dummy number to change when needing to force rebuild without changing the definition: 1
