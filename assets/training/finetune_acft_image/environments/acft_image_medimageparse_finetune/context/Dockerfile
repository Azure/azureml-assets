FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update

COPY requirements.txt .

# Seed modern numpy
RUN pip install numpy==1.26.4

# Install azureml-automl-core without build isolation so it uses already-installed numpy
RUN pip install --no-build-isolation azureml-automl-core==1.60.0 

RUN pip install -r requirements.txt