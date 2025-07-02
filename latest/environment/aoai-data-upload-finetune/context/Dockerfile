FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04@sha256:7c4531f08fa579b70be615a8df2594cb7f5d303f70993816e3c1f119bc692ac9

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir