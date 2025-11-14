FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
COPY tracking.py /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/tracking.py
COPY numeric_math_verify.py /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/numeric_math_verify.py
COPY __init__.py /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/utils/reward_score/__init__.py
COPY base_model_merger.py /opt/conda/envs/ptca/lib/python3.10/site-packages/verl/model_merger/base_model_merger.py

# Install packages with special build requirements separately
RUN pip install vllm
RUN MAX_JOBS=4 pip install --upgrade flash-attn --no-build-isolation

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

RUN pip list