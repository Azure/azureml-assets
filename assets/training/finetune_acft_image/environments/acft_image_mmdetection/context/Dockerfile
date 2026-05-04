# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
RUN apt-get update && apt-get install -y expat && apt-get clean && rm -rf /var/lib/apt/lists/*
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

# upgrade pip, wheel, setuptools and transitive dependencies in ptca env
# NOTE: azureml-mlflow~=1.62.0.post2 pins cryptography<47.0.0; upgrading anyway for CVE fix
# protobuf is required by azureml-acft-accelerator but it is not pinned hence upgrade
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5 setuptools==82.0.0 cryptography==46.0.7 pillow==12.2.0 \
    'requests>=2.33.0' 'aiohttp>=3.13.4' 'onnx>=1.21.0' 'pytest>=9.0.3'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y 'pip>=26.0.1' 'wheel>=0.46.2'
# vulnerability in base conda env
# python-dotenv 1.2.1 (GHSA-mf9w-mj56-hr94): brought in by azureml-inference-server-http -> pydantic-settings -> python-dotenv>=0.21.0;
#   parent packages do not upper-bound python-dotenv so upgrading them won't force >=1.2.2; direct override required (checked 2026-05-04)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'pip>=26.0.1' 'wheel>=0.46.2' 'setuptools>=82.0.1' \
    cryptography==46.0.7 'requests>=2.33.0' 'aiohttp>=3.13.4' 'python-dotenv>=1.2.2'
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
