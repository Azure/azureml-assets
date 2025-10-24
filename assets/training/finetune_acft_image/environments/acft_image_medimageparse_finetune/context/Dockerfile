FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update

COPY requirements.txt .

RUN pip install -r requirements.txt
# Security vulnerability fix - GHSA-4xh5-x5gv-qwph: Upgrade pip
RUN pip install --upgrade pip
RUN /opt/conda/bin/pip install --upgrade pip || true
RUN /opt/conda/envs/ptca/bin/pip install --upgrade pip || true
RUN /opt/miniconda/bin/pip install --upgrade pip || true


