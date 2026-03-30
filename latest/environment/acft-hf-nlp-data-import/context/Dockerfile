# openmpi image
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260329.v1

USER root

# sudo is expected by Singularity inside the image
# Security: upgrade all OS packages and install security-patched system libraries
RUN apt-get update && ACCEPT_EULA=Y apt-get -y upgrade && apt-get install -y sudo libxml2 sqlite3 libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl 

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

# pip==26.0.1, setuptools==82.0.1, wheel==0.46.3, cryptography==46.0.5, urllib3==2.6.3, h2==4.3.0
# are already at fixed versions in the openmpi base image (20260315.v1).
# The override below only targets packages NOT fixed in base or pulled in vulnerable by requirements.txt.
RUN pip install --no-cache-dir scikit-learn==1.5.1 aiohttp==3.13.3 && rm -rf /root/.cache/pip

# The below file is required for baking the code into the environment 
COPY data_import_run.py /azureml/data_import/run.py

# dummy number to change when needing to force rebuild without changing the definition: 2

