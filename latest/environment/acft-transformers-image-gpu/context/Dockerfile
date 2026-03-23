# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202603.2

USER root
RUN apt-get -y update && apt-get -y upgrade

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel, protobuf vulnerabilities in ptca env
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# protobuf parent is mlflow==3.5.0 (already latest) which accepts protobuf>=3.12.0, cannot force 6.33.5 via parent
# pyasn1 is a transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
# parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 protobuf==6.33.5 cryptography==46.0.5 pyasn1==0.6.3 pillow==12.1.1 fastmcp>=2.14.0
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y pip>=26.0.1 wheel>=0.46.2
# vulnerability in base conda env
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip>=26.0 wheel>=0.46.2 setuptools>=82.0.0 cryptography==46.0.5 'PyJWT>=2.12.0'
