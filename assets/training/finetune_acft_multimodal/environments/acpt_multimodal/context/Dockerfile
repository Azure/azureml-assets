# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch222:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get install -y expat

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Downgraded azureml-metrics, to prevent bug related to logger in azureml-metrics. Should be upgraded in next pypi release of acft-multimodal
# Removing it from requirements.txt to avoid conflict with scikit-learn version
RUN pip install azureml-metrics==0.0.33

# Vulnerability fix
RUN pip install datasets==2.15.0
RUN pip install pyarrow==14.0.1
RUN pip install cryptography==44.0.1
RUN pip install gunicorn==22.0.0
RUN pip install onnxruntime==1.17.3
RUN pip install onnx==1.17.0
RUN pip install scikit-learn==1.5.0