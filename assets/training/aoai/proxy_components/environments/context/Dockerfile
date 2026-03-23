FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

# Security vulnerability fixes
RUN pip install --no-cache-dir --upgrade 'pyasn1>=0.6.3'
