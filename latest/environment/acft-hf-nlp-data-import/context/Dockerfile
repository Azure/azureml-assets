# openmpi image
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20250619.v1

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
