# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# openmim still imports pkg_resources, which is removed from setuptools>=82.
# Keep a temporary compatibility pin for mim installation, then restore setuptools.
RUN pip install --no-cache-dir 'setuptools<82'

# # # Install mmdet
# # Note that MMDet installs pycocotools
# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==3.3.0
# Temporary workaround for https://github.com/open-mmlab/mmdetection/issues/11668 (when mmdet updated, remove lines below)
RUN mim install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html --no-cache-dir
RUN pip install --no-cache-dir --upgrade setuptools==82.0.0
RUN sed -i 's/2.2.0/2.3.0/' /opt/conda/envs/ptca/lib/python3.10/site-packages/mmdet/__init__.py

# requests==2.32.4 in requirements.txt downgrades the base-image version; re-upgrade for security
# onnx: azureml-acft-accelerator pins onnx<=1.17.0 but that range has known CVEs;
#   onnx-weekly 1.22.0.dev20260504 includes main-branch security fixes (GHSA-3r9x-f23j-gc73,
#   GHSA-538c-55jv-c5g9, GHSA-hqmj-h5c6-369m, etc.) not yet in a stable PyPI release (checked 2026-05-04)
RUN pip install --no-cache-dir --upgrade 'requests>=2.33.0' && \
    pip uninstall -y onnx && pip install --no-cache-dir 'onnx-weekly>=1.22.0.dev20260504'
# vulnerability in base conda env
# python-dotenv 1.2.1 (GHSA-mf9w-mj56-hr94): brought in by azureml-inference-server-http -> pydantic-settings -> python-dotenv>=0.21.0;
#   parent packages do not upper-bound python-dotenv so upgrading them won't force >=1.2.2; direct override required (checked 2026-05-04)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
