FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py39-torch1131:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

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
RUN pip install -r requirements.txt --no-cache-dir

## Vulnerability fix
RUN pip install Pillow gunicorn onnx idna tqdm requests==2.32.0 tornado==6.4.1 certifi==2024.07.04 urllib3==1.26.19 scikit-learn==1.5.0
# List installed packages
RUN pip list

## Delete
RUN rm requirements.txt
