# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202604.1

USER root
RUN apt-get -y update && apt-get -y upgrade
# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Downgraded azureml-metrics, to prevent bug related to logger in azureml-metrics. Should be upgraded in next pypi release of acft-multimodal
# Removing it from requirements.txt to avoid conflict with scikit-learn version
RUN pip install azureml-metrics==0.0.33 pyarrow==14.0.1

RUN pip install azureml-acft-common-components==0.0.89
RUN pip install azureml-acft-accelerator==0.0.89

# protobuf is a transitive dep of onnxruntime-training; base image has older version pre-installed
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# setuptools vendors jaraco.context internally; >=82.0.1 bundles the patched version (GHSA-58pv-8j8x-9vj2)
# aiohttp: transitive dep of azure-core/datasets; parents use loose floors (GHSA-mwh4-6h8g-pg8w etc.)
# onnx: onnxruntime-training accepts onnx>=1.16.0; override needed (GHSA-p433-9wv8-28xj etc.)
# nltk: GHSA-gfwx-w7gr-fvh7; >=3.9.4 required
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5 cryptography==46.0.7 pillow==12.1.1 setuptools>=82.0.1 'requests>=2.33.0' 'aiohttp>=3.13.4' 'onnx>=1.21.0' 'nltk>=3.9.4'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -y -n ptca pip>=26.0.1 wheel>=0.46.2 
# vulnerability in base conda env
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip>=26.0.1 wheel>=0.46.2 setuptools>=82.0.1 cryptography==46.0.7 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'requests>=2.33.0'
