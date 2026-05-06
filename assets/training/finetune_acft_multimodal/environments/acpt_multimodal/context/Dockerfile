# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade
# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Downgraded azureml-metrics, to prevent bug related to logger in azureml-metrics. Should be upgraded in next pypi release of acft-multimodal
# Removing it from requirements.txt to avoid conflict with scikit-learn version
RUN pip install azureml-metrics==0.0.33 pyarrow==14.0.1

RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install azureml-acft-accelerator=={{latest-pypi-version}}

# setuptools: >=82.0.1 patches jaraco.context (GHSA-58pv-8j8x-9vj2); base ships 82.0.0
# nltk: >=3.9.4 required (GHSA-gfwx-w7gr-fvh7)
# transformers: ensure >=5.0.0 after azureml-* installs (GHSA-69w3-r845-3855)
# onnx: onnxruntime-training has loose onnx dep; pip resolves to 1.17.0 which has multiple CVEs
RUN pip install --no-cache-dir --upgrade 'setuptools>=82.0.1' 'nltk>=3.9.4' 'transformers>=5.0.0' 'onnx>=1.21.0'
# base conda env: setuptools and python-dotenv need upgrades above what base image ships
# python-dotenv: transitive dep of mlflow-skinny (accepts >=0.19.0,<2); base ships 1.2.1 (GHSA-mf9w-mj56-hr94)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'setuptools>=82.0.1' 'python-dotenv>=1.2.2'
