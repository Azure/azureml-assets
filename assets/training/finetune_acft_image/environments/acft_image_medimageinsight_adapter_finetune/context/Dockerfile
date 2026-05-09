# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade

RUN apt-get -y install unzip

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Transitive dep overrides where pip may not resolve to the patched version:
# pyasn1: mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1;
#   parents use loose floors so pip may resolve to <0.6.3 (CVE-2026-30922)
# Mako: mlflow → alembic → Mako; alembic uses unpinned Mako dep
# python-dotenv: mlflow → mlflow-skinny → python-dotenv<2,>=0.19.0;
#   mlflow 3.11.1 (latest as of 2026-05-04) allows >=1.2.2 but pip may resolve older (GHSA-mf9w-mj56-hr94)
RUN pip install --no-cache-dir --upgrade 'pyasn1>=0.6.3' 'Mako>=1.3.11' 'python-dotenv>=1.2.2'
# python-dotenv in /opt/conda Python 3.13 also ships as vulnerable 1.2.1 in the base image.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/