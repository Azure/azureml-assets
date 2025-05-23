FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch222:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

WORKDIR /

RUN apt-get update && apt-get upgrade -y && apt-get install wget nscd python3-idna git git-lfs libxml2 -y

# azcopy install starts
RUN echo "Downloading azcopy to file azcopy.tar ....";\
    wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar --no-verbose;\
    tar -xvf azcopy.tar;\
    mkdir -p ./azcopy/bin/;\
    cp ./azcopy_linux_amd64_*/azcopy ./bin/;\
    rm azcopy.tar;\
    rm -rf ./azcopy_linux_amd64_*;\
    which azcopy | grep -o azcopy > /dev/null &&  echo "azcopy not installed" || echo "azcopy installed";

COPY requirements.txt .

# Switch to python==3.9 to prevent numpy==1.23.5 build failure
# numpy==1.23.5 is a hard dependency in azureml-automl-dnn-vision
# which is not compatible with py310.
# Going forward  azureml-automl-dnn-vision needs to update version for torch and numpy
RUN conda install python=3.9 -y

RUN pip install -r requirements.txt --no-cache-dir

# Vulnerability fix
# Remove Conda-installed Torch and related packages
RUN conda remove -y --force \
    pytorch \
    libtorch \
    torchaudio \
    torchvision || true

# Remove old Conda cached torch build (optional but clean)
RUN rm -rf /opt/conda/pkgs/pytorch-* \
    && rm -rf /opt/conda/pkgs/libtorch-* \
    && rm -rf /opt/conda/pkgs/torchaudio-* \
    && rm -rf /opt/conda/pkgs/torchvision-*

# Clean pip cache (optional)
RUN pip cache purge

# Install Torch 2.6.0 and compatible torchvision
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0

# List pip packages
RUN pip list

## Delete
RUN rm requirements.txt
