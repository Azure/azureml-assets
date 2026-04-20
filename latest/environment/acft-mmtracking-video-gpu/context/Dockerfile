# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202604.2

USER root
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y expat

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir


# [Temp Fix] To address PyTorch Distributed RPC Framework Remote Code Execution (RCE) Vulnerability - Working internally to get a compatbile image, will re-enable
# RUN mim install mmtrack==0.14.0
# RUN mim install mmcv-full==1.7.1
# # # Note that MMDet installs pycocotools
# # Note: mmdet should be installed via mim to access the model zoo config folder.
# RUN mim install mmdet==2.28.2

# vulnerability fixes - this will be removed once we update to MMTracking's latest version
RUN pip install scipy==1.10.1
RUN pip install gunicorn==23.0.0
RUN pip install onnx==1.21.0

# It is recommended to upgrade to numpy version 1.23.5 to address the numpy.core.multiarray issue
RUN pip install numpy==1.23.5

# pin required version of yapf to avoid breaking changes in newer versions
# https://github.com/open-mmlab/mmdetection/issues/10962
RUN pip install yapf==0.40.1

# protobuf is a transitive dep of onnx/mlflow-skinny; parents use loose floors (>=3.12.0), cannot force 6.33.5
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# aiohttp: transitive dep of azure-core/mlflow; parents use loose floors (GHSA-mwh4-6h8g-pg8w etc.)
# requests: transitive dep of azure-core/mlflow; parents use loose floors (GHSA-gc5v-m9x4-r6x2)
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5 cryptography==46.0.7 pillow==12.1.1 'aiohttp>=3.13.4' 'requests>=2.33.0'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -y -n ptca pip>=26.0.1 wheel>=0.46.2 
# Upgrade requests, urllib3, aiohttpin the system Python (3.13) for fixing vulnerability
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade pip>=26.0.1 wheel>=0.46.2 setuptools>=82.0.1 cryptography==46.0.7 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'requests>=2.33.0'
