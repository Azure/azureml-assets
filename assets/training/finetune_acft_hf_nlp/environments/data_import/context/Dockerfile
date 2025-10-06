# openmpi image
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

USER root

# sudo is expected by Singularity inside the image
RUN apt-get update && ACCEPT_EULA=Y apt-get -y upgrade && apt-get install sudo

# --- Security Vulnerability Fixes ---
# Fix dpkg vulnerability (USN-7768-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    dpkg-dev=1.22.6ubuntu6.5 \
    dpkg=1.22.6ubuntu6.5 \
    libdpkg-perl=1.22.6ubuntu6.5

# Fix GNU C Library vulnerability (USN-7760-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libc-bin=2.39-0ubuntu8.6 \
    libc6-dev=2.39-0ubuntu8.6 \
    libc-dev-bin=2.39-0ubuntu8.6 \
    libc6=2.39-0ubuntu8.6

# Fix libxml2 vulnerability (USN-7743-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libxml2=2.9.14+dfsg-1.3ubuntu3.5

# Fix PAM vulnerability (USN-7761-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libpam-runtime=1.5.3-5ubuntu5.5 \
    libpam0g=1.5.3-5ubuntu5.5 \
    libpam-modules-bin=1.5.3-5ubuntu5.5 \
    libpam-modules=1.5.3-5ubuntu5.5

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

# temp fix for vulnerabilties, remove after pypi release
RUN pip install scikit-learn==1.5.1

# The below file is required for baking the code into the environment 
COPY data_import_run.py /azureml/data_import/run.py

# dummy number to change when needing to force rebuild without changing the definition: 2
