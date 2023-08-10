# PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131:{{latest-image-tag:vision\.biweekly\.\d{6}\.\d+}}

USER root
RUN apt-get -y update

RUN pip install azureml-metrics==0.0.24
RUN pip install azureml-evaluate-mlflow==0.0.24
RUN pip install azureml-acft-accelerator==0.0.24
RUN pip install azureml-acft-common-components["image"]==0.0.24
RUN pip install azureml-acft-image-components==0.0.24

# azureml-core 1.50 is failing the run due to azureml-dataprep package. Further inverstigation needed. The source of
# azureml-core is azurem-evaluate-mlflow and azureml-acft-common-component package.
RUN pip install azureml-core==1.49.0

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# diffusers >= 0.16 is requirement of the base image
RUN pip install diffusers==0.17.1
