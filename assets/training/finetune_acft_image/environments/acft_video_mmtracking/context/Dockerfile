# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir


# [Temp Fix] To address PyTorch Distributed RPC Framework Remote Code Execution (RCE) Vulnerability - Working internally to get a compatbile image, will re-enable
# RUN mim install mmtrack==0.14.0
# RUN mim install mmcv-full==1.7.1
# # # Note that MMDet installs pycocotools
# # Note: mmdet should be installed via mim to access the model zoo config folder.
# RUN mim install mmdet==2.28.2

# Override transformers to fix GHSA-69w3-r845-3855 (CVE-2026-1839, arbitrary code execution in Trainer)
# Root cause: azureml-acft-image-components pins transformers==4.53.0; cannot upgrade parent
# Using --no-deps to avoid pulling incompatible transitive deps for the older HF stack
RUN pip install --no-cache-dir --no-deps 'transformers==5.5.4'

# vulnerability fixes - this will be removed once we update to MMTracking's latest version
RUN pip install scipy==1.10.1
RUN pip install gunicorn==23.0.0
RUN pip install onnx==1.21.0

# It is recommended to upgrade to numpy version 1.23.5 to address the numpy.core.multiarray issue
RUN pip install numpy==1.23.5

# pin required version of yapf to avoid breaking changes in newer versions
# https://github.com/open-mmlab/mmdetection/issues/10962
RUN pip install yapf==0.40.1

# python-dotenv 1.2.1 in system Python (3.13) needs upgrade to 1.2.2 (GHSA-mf9w-mj56-hr94, symlink overwrite);
# transitive dep of mlflow-skinny (>=0.19.0,<2) and pydantic-settings (>=0.21.0) — parents use loose floors
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
