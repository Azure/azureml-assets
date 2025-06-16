FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch260:biweekly.202506.2

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

# This line is added because azureml-automl-dnn-vision depends on python 3.9.
# Added ignore python requires to get this installed.
RUN pip install --ignore-requires-python azureml-automl-dnn-vision

RUN pip install -r requirements.txt --no-cache-dir

# Adding this to exclude Python 3.13 packages from the setuptools library fix
RUN rm -rf /opt/conda/lib/python3.13 || true

# List pip packages
RUN pip list

## Delete
RUN rm requirements.txt
