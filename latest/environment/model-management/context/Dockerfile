FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu113-py38-torch1110@sha256:29f2aaee2bb78918998fa9908e3e95b72fa5b6f2898667379d7584596526b902

WORKDIR /

RUN apt-get update && apt-get install -y git git-lfs

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

# List installed packages
RUN pip list

## Delete
RUN rm requirements.txt
