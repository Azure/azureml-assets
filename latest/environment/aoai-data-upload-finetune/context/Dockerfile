FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04@sha256:a80c5cf13d0e829481fb3d28bdd768a36021410ba039987cfa86f3efcb3068d8

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install mlflow==3.1.0