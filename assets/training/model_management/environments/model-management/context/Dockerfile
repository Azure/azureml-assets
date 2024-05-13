FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py39-torch1131:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

WORKDIR /

RUN apt-get update && apt-get upgrade -y

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
RUN pip install Pillow==10.3.0 gunicorn==22.0.0 onnx==1.16.0 tqdm==4.66.3 idna==3.7 Jinja2==3.1.4 Werkzeug==3.0.3
# List installed packages
RUN pip list

## Delete
RUN rm requirements.txt
