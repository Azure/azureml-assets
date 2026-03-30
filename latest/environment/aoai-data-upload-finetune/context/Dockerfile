FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260329.v1

# Security: upgrade all OS packages to patch USN vulnerabilities
RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
