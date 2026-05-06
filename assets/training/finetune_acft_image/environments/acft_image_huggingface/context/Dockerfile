# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Override vulnerable transitive deps in ptca env that are not fixed in the base image
# onnx: azureml-acft-accelerator==0.0.89 requires onnx<=1.17.0 which downgrades base 1.21.0;
#   override needed to keep safe version (GHSA-p433-9wv8-28xj etc.)
# pyasn1: transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
#   parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
# fastmcp: transitive dep (mlflow-skinny[mcp] requires fastmcp<4,>=2.0.0); loose floor,
#   GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767; override to >=3.2.0
# Mako: transitive dep (mlflow → alembic → Mako); alembic 1.18.4 requires Mako (no version pin),
#   pip won't upgrade pre-installed Mako; override needed (GHSA-v92g-xgxw-vvmm)
# GitPython: transitive dep (mlflow → mlflow-skinny requires gitpython<4,>=3.1.9); loose floor,
#   pip resolves to 3.1.46 which has GHSA-rpm5-65cw-6hj4, GHSA-x2qx-6953-8485; override to >=3.1.47
# python-dotenv: transitive dep (mlflow → mlflow-skinny requires python-dotenv<2,>=0.19.0); loose floor,
#   pip resolves to 1.2.1 which has GHSA-mf9w-mj56-hr94; override to >=1.2.2
RUN pip install --no-cache-dir --upgrade 'onnx>=1.21.0' pyasn1==0.6.3 'fastmcp>=3.2.0' Mako==1.3.11 'GitPython>=3.1.47' 'python-dotenv>=1.2.2'
# python-dotenv in base conda env: transitive dep of uvicorn[standard] (>=0.13); loose floor,
#   base image has 1.2.1 which has GHSA-mf9w-mj56-hr94; override to >=1.2.2
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
