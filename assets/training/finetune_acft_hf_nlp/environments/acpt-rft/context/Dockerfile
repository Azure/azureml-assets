FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root
RUN apt-get update && apt-get -y upgrade
RUN pip install pip==25.3
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
COPY tracking /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py

RUN MAX_JOBS=4 pip install --upgrade flash-attn==2.8.3 --no-build-isolation 
RUN pip install sglang==0.5.4
RUN pip install sgl-kernel==0.3.16.post3
RUN pip install starlette==0.49.1
RUN pip install mlflow==3.0.0
# clean conda and pip caches
RUN rm -rf ~/.cache/pip