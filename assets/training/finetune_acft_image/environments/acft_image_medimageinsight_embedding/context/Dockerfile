# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

# Install unzip and upgrade OS packages to fix vulnerabilities
RUN apt-get -y update && apt-get -y upgrade && apt-get -y install unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel and transitive deps to fix vulnerabilities
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# protobuf is a transitive dep of mlflow-skinny/onnx; parents use loose floors (>=3.12.0), cannot force 6.33.5
# mlflow 3.5.0 has CVEs (CVE-2025-14287, CVE-2026-2033, CVE-2026-2635); upgrade after requirements install
# azureml-mlflow pins mlflow-skinny<=3.5.0, so mlflow must be upgraded separately to avoid resolution conflict
RUN pip install --no-cache-dir mlflow==3.10.1
# pyasn1 is a transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
# parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
RUN pip install --no-cache-dir --upgrade protobuf==6.33.5 cryptography==46.0.6 pyasn1==0.6.3 pillow==12.1.1 wheel>=0.46.2 'fastmcp>=3.2.0' 'onnx>=1.21.0' 'requests>=2.33.0'

# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -y -n ptca pip>=26.0.1 wheel>=0.46.2 

# Upgrade packages in the system Python(3.13) for fixing vulnerability
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade urllib3==2.6.3 aiohttp==3.13.4 PyNaCl==1.6.2 pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.6 'PyJWT>=2.12.0' 'requests>=2.33.0'
