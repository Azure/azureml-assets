# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch201:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Vulnerability fix
RUN pip install datasets==2.15.0
RUN pip install pyarrow==14.0.1
RUN pip install cryptography==42.0.5
RUN pip install gunicorn==22.0.0
RUN pip install onnxruntime==1.17.3
RUN pip install onnx==1.16.0 
# dummy number to change when needing to force rebuild without changing the definition: 1