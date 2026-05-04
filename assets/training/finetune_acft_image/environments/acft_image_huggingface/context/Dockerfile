# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel, and override vulnerable transitive deps in ptca env
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# protobuf: mlflow-skinny==3.11.1 requires protobuf<8,>=3.12.0 (loose floor); override needed
# pyasn1: transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
#   parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
# aiohttp: transitive dep of azure-core/datasets; parents use loose floors (GHSA-mwh4-6h8g-pg8w etc.)
# onnx: onnxruntime/azureml-acft-accelerator require onnx>=1.16.0; override needed (GHSA-p433-9wv8-28xj etc.)
# fastmcp: GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767; >=3.2.0 required
# requests: transitive dep of azure-core/mlflow-skinny; parents use loose floors (GHSA-gc5v-m9x4-r6x2)
# Mako: transitive dep (mlflow → alembic → Mako); alembic 1.18.4 requires Mako (no version pin),
#   pip won't upgrade pre-installed Mako; override needed (GHSA-v92g-xgxw-vvmm)
# pytest: transitive dep from azureml packages; parent packages use loose floors so pip resolves
#   to 7.4.3 which has CVE-2025-71176 (GHSA-6w46-j5rx-g56g); override to >=9.0.3
# GitPython: transitive dep (mlflow → mlflow-skinny requires gitpython<4,>=3.1.9); loose floor,
#   pip resolves to 3.1.46 which has GHSA-rpm5-65cw-6hj4, GHSA-x2qx-6953-8485; override to >=3.1.47
# python-dotenv: transitive dep (mlflow → mlflow-skinny requires python-dotenv<2,>=0.19.0); loose floor,
#   pip resolves to 1.2.1 which has GHSA-mf9w-mj56-hr94; override to >=1.2.2
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5 cryptography==46.0.7 pyasn1==0.6.3 pillow==12.2.0 'fastmcp>=3.2.0' 'aiohttp>=3.13.4' 'onnx>=1.21.0' 'requests>=2.33.0' Mako==1.3.11 'pytest>=9.0.3' 'GitPython>=3.1.47' 'python-dotenv>=1.2.2'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y pip>=26.0.1 wheel>=0.46.2
# vulnerability in base conda env
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
# python-dotenv: transitive dep of uvicorn[standard] (>=0.13) in base env; loose floor,
#   pip resolves to 1.2.1 which has GHSA-mf9w-mj56-hr94; override to >=1.2.2
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip>=26.0 wheel>=0.46.2 setuptools>=82.0.0 cryptography==46.0.7 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'requests>=2.33.0' 'python-dotenv>=1.2.2'
