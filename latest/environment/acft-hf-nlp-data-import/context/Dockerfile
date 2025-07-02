# openmpi image
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04@sha256:7c4531f08fa579b70be615a8df2594cb7f5d303f70993816e3c1f119bc692ac9

USER root

# sudo is expected by Singularity inside the image
RUN apt-get update && ACCEPT_EULA=Y apt-get -y upgrade && apt-get install sudo
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

# temp fix for vulnerabilties, remove after pypi release
RUN pip install scikit-learn==1.5.1

# The below file is required for baking the code into the environment 
COPY data_import_run.py /azureml/data_import/run.py

# dummy number to change when needing to force rebuild without changing the definition: 2
