#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --only-upgrade linux-headers-5.15.0-163-generic=5.15.0-163.173 linux-headers-5.15.0-163=5.15.0-163.173 linux-libc-dev=5.15.0-163.173 || \
    apt-get install -y --only-upgrade linux-headers-generic linux-libc-dev && \
    apt-get install -y --only-upgrade git tar binutils binutils-common binutils-x86-64-linux-gnu libbinutils wget && \
    apt-get autoremove -y linux-headers-5.15.0-153 linux-headers-5.15.0-153-generic linux-headers-5.15.0-161 linux-headers-5.15.0-161-generic 2>/dev/null || true && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install pip==25.3
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}

# following are for vulnerability overrides at later\
# release of following packages consider moving then to requirements.txt
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
RUN pip install --no-cache-dir --upgrade "wandb>=0.23.0"
RUN pip install xgrammar==0.1.27
RUN pip install transformers==4.57.1
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
RUN rm -rf /opt/conda/lib/python3.10/site-packages/pip-25.2.dist-info
RUN rm -rf opt/conda/lib/python3.13/site-packages/pip-25.2.dist-info
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
