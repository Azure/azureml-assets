# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202512.1

USER root
RUN apt-get -y update && apt-get install -y expat libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Downgraded azureml-metrics, to prevent bug related to logger in azureml-metrics. Should be upgraded in next pypi release of acft-multimodal
# Removing it from requirements.txt to avoid conflict with scikit-learn version
RUN pip install azureml-metrics==0.0.33 pyarrow==14.0.1

RUN pip install azureml-acft-common-components==0.0.84
RUN pip install azureml-acft-accelerator==0.0.84
