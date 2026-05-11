# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.1

USER root
# openssh-client is reinstalled explicitly after `apt-get upgrade` to force pickup
# of USN-8222-1 (>= 1:8.9p1-3ubuntu0.15). The stable-ubuntu2204-cu126-py310-torch280
# base image (biweekly.202605.1) layer pins/holds the older 1:8.9p1-3ubuntu0.14 and
# `apt-get upgrade` alone does not bump it. openssh is shipped by the Ubuntu base
# (no parent package in this context), so the only fix is the apt upgrade itself.
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install --reinstall -y openssh-client && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Downgraded azureml-metrics, to prevent bug related to logger in azureml-metrics. Should be upgraded in next pypi release of acft-multimodal
# Removing it from requirements.txt to avoid conflict with scikit-learn version
RUN pip install azureml-metrics==0.0.33 pyarrow==14.0.1

RUN pip install azureml-acft-common-components==0.0.89
RUN pip install azureml-acft-accelerator==0.0.89

# setuptools: >=82.0.1 patches jaraco.context (GHSA-58pv-8j8x-9vj2); base ships 82.0.0
# nltk: >=3.9.4 required (GHSA-gfwx-w7gr-fvh7)
# transformers: ensure >=5.0.0 after azureml-* installs (GHSA-69w3-r845-3855)
# onnx: onnxruntime-training has loose onnx dep; pip resolves to 1.17.0 which has multiple CVEs
RUN pip install --no-cache-dir --upgrade 'setuptools>=82.0.1' 'nltk>=3.9.4' 'transformers>=5.0.0' 'onnx>=1.21.0'
# base conda env: setuptools and python-dotenv need upgrades above what base image ships
# python-dotenv: transitive dep of mlflow-skinny (accepts >=0.19.0,<2); base ships 1.2.1 (GHSA-mf9w-mj56-hr94)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'setuptools>=82.0.1' 'python-dotenv>=1.2.2'
