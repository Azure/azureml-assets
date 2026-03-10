FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade "pip>=26.0"

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install mlflow==3.5.0 "starlette>=0.49.1"

# Security vulnerability fixes
RUN pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 'Werkzeug>=3.1.5' 'azure-core>=1.38.0' 'jaraco.context>=6.1.0' 'protobuf>=6.33.5' 'pyasn1>=0.6.2' 'pillow>=12.1.1'