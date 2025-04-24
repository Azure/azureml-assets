# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch222:biweekly.202504.2

RUN apt-get -y update && apt-get install -y expat

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# Vulnerability Fix
RUN pip install gunicorn==23.0.0
RUN pip install certifi==2024.07.04
RUN pip install onnx==1.17.0