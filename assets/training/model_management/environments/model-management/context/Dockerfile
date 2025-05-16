FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch222:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

WORKDIR /

RUN apt-get update && apt-get upgrade -y && apt-get install wget nscd python3-idna git git-lfs -y

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
RUN pip install torch==2.6.0 torchvision --no-cache-dir

# List pip packages
RUN pip list

## Delete
RUN rm requirements.txt
