#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202603.1

USER root
RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow==0.0.87
RUN pip install azureml-acft-common-components==0.0.87
RUN pip install transformers==4.56.0

# mlflow 3.5.0 has CVEs (CVE-2025-14287, CVE-2026-2033, CVE-2026-2635); upgrade after azureml packages
# azureml-evaluate-mlflow → azureml-mlflow pins mlflow-skinny<=3.5.0, conflicting with mlflow 3.10.1
# so mlflow must be upgraded separately to avoid pip resolution conflict
RUN pip install --no-cache-dir mlflow==3.10.1

# upgrade pip, wheel, setuptools and transitive deps to fix vulnerabilities
# protobuf is a transitive dep of vllm; parent uses loose floors, cannot force 6.33.5
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# pyasn1 is a transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
# parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
RUN pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 protobuf==6.33.5 cryptography==46.0.5 pyasn1==0.6.3 python-multipart==0.0.22 urllib3==2.6.3 azure-core==1.38.0 xgrammar>=0.1.32
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y pip>=26.0.1 wheel>=0.46.2
# vulnerability in base conda env
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.5 'PyJWT>=2.12.0'
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
