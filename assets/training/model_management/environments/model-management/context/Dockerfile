FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch260:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

WORKDIR /

RUN apt-get update && apt-get upgrade -y && apt-get install wget nscd python3-idna git git-lfs libxml2 -y

# ✅ Fix Go stdlib CVEs by installing Go 1.24.4
RUN rm -rf /usr/local/go && \
    wget https://go.dev/dl/go1.24.4.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz && \
    rm go1.24.4.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:$PATH"

# ✅ Secure AzCopy install (built with Go >=1.24.4)
RUN echo "Downloading secure AzCopy..." && \
    wget https://azcopyvnext.azureedge.net/release20240702/azcopy_linux_amd64_10.24.0.tar.gz -O azcopy.tar.gz && \
    tar -xzf azcopy.tar.gz && \
    cp ./azcopy_linux_amd64_*/azcopy /usr/bin/azcopy && \
    chmod +x /usr/bin/azcopy && \
    rm -rf azcopy.tar.gz azcopy_linux_amd64_*

COPY requirements.txt .

# This line is added because azureml-automl-dnn-vision depends on python 3.9.
# Added ignore python requires to get this installed.
# RUN pip install --ignore-requires-python azureml-automl-dnn-vision
RUN pip install azureml-automl-dnn-vision

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install --upgrade nltk==3.9

# Adding this to exclude Python 3.13 packages from the setuptools library fix
RUN rm -rf /opt/conda/lib/python3.13 || true

# List pip packages
RUN pip list

# Cleanup
RUN rm requirements.txt
