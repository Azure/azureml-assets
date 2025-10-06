FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

RUN apt-get update && apt-get -y upgrade

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

# Fix PAM vulnerability (USN-7761-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \
    libpam-runtime=1.5.3-5ubuntu5.5 \
    libpam0g=1.5.3-5ubuntu5.5 \
    libpam-modules-bin=1.5.3-5ubuntu5.5 \
    libpam-modules=1.5.3-5ubuntu5.5

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install mlflow==3.1.0