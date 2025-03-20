FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch222:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

WORKDIR /

# Update package list, upgrade system, and install required packages in one step
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    wget nscd python3-idna git git-lfs \
    libctf-nobfd0=2.38-4ubuntu2.7 \
    libctf0=2.38-4ubuntu2.7 \
    libbinutils=2.38-4ubuntu2.7 \
    binutils-common=2.38-4ubuntu2.7 \
    binutils-x86-64-linux-gnu=2.38-4ubuntu2.7 \
    binutils=2.38-4ubuntu2.7 \
    libxml2=2.9.13+dfsg-1ubuntu0.6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

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
RUN pip install Pillow gunicorn onnx==1.17.0 idna tqdm requests==2.32.1 tornado==6.4.2 certifi==2024.07.04 urllib3==1.26.19 scikit-learn==1.5.1 mlflow==2.19 mlflow-skinny==2.19 marshmallow==3.23.2

# List pip packages
RUN pip list

## Delete
RUN rm requirements.txt
