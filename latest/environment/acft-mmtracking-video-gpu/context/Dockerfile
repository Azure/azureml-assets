# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202507.1

USER root
RUN apt-get -y update && apt-get install -y expat

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# # # Install mmcv-full and mmdet
# Request for scipy installation to avoid error in mmcv-full installation
RUN apt-get install -y gfortran libopenblas-dev liblapack-dev

# [Temp Fix] To address PyTorch Distributed RPC Framework Remote Code Execution (RCE) Vulnerability - Working internally to get a compatbile image, will re-enable
# RUN mim install mmtrack==0.14.0
# RUN mim install mmcv-full==1.7.1
# # # Note that MMDet installs pycocotools
# # Note: mmdet should be installed via mim to access the model zoo config folder.
# RUN mim install mmdet==2.28.2

# vulnerability fixes - this will be removed once we update to MMTracking's latest version
RUN pip install scipy==1.10.1
RUN pip install gunicorn==23.0.0
RUN pip install onnx==1.17.0

# It is recommended to upgrade to numpy version 1.23.5 to address the numpy.core.multiarray issue
RUN pip install numpy==1.23.5

# pin required version of yapf to avoid breaking changes in newer versions
# https://github.com/open-mmlab/mmdetection/issues/10962
RUN pip install yapf==0.40.1

# Upgrade requests in the system Python (3.13) for fixing vulnerability
RUN /opt/conda/bin/python3.13 -m pip install --upgrade requests || true
