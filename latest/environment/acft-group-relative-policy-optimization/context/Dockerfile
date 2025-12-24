#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202512.2

USER root

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow==0.0.84
RUN pip install azureml-acft-common-components==0.0.84
RUN pip install transformers==4.56.0

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

# vulnerabilities
RUN pip install --upgrade --no-cache-dir 'urllib3>=2.6.0'
