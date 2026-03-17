#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install transformers==4.56.0

# upgrade pip, wheel, setuptools and transitive deps to fix vulnerabilities
# protobuf is a transitive dep of vllm; parent uses loose floors, cannot force 6.33.5
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
RUN pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 protobuf==6.33.5 cryptography==46.0.5 pyasn1==0.6.2 python-multipart==0.0.22 urllib3==2.6.3 azure-core==1.38.0
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y pip>=26.0.1 wheel>=0.46.2
# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.5
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
