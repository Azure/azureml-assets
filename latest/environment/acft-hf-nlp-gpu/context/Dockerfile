#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch222:biweekly.202404.3

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
# The below file is required for baking the code into the environment
COPY finetune_run.py /azureml/finetune/run.py

RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
RUN MAX_JOBS=4 pip install flash-attn==2.5.5 --no-build-isolation

# dummy number to change when needing to force rebuild without changing the definition: 1
