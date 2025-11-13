FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install azureml-acft-common-components==0.0.77
RUN pip install azureml-evaluate-mlflow==0.0.77
COPY tracking.py /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py
COPY numeric_math_verify.py /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/numeric_math_verify.py
COPY __init__.py /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/__init__.py
COPY base_model_merger.py /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/model_merger/base_model_merger.py

RUN pip install transformers==4.56.1
RUN pip install vllm==0.10.2
RUN MAX_JOBS=4 pip install --upgrade flash-attn==2.8.3 --no-build-isolation
RUN pip install click==8.2.1
RUN pip install sglang==0.5.2
RUN pip install sgl-kernel==0.3.14
RUN pip install math-verify==0.8.0
# clean conda and pip caches
RUN rm -rf ~/.cache/pip