FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py39-torch1131:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

WORKDIR /

<<<<<<< HEAD
RUN apt-get update && apt-get upgrade -y && apt-get install curl binutils libpam-modules libgnutls30 libssh-4 libsqlite3-0 openssh-client openssh-server git git-lfs -y
=======
RUN apt-get update && apt-get upgrade -y && apt-get install nscd python3-idna git git-lfs -y
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa

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
<<<<<<< HEAD
RUN pip install Pillow==10.2.0

=======
RUN pip install Pillow gunicorn onnx idna tqdm requests==2.32.0 tornado==6.4.1
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
# List installed packages
RUN pip list

## Delete
RUN rm requirements.txt
# dummy number to change when needing to force rebuild without changing the definition: 1
