FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
WORKDIR /

RUN apt-get update && apt-get upgrade -y && apt-get install wget nscd python3-idna git git-lfs libxml2 -y

RUN wget https://go.dev/dl/go1.24.9.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.24.9.linux-amd64.tar.gz && \
    rm go1.24.9.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:$PATH"

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip==25.3 setuptools wheel

RUN /opt/conda/envs/ptca/bin/pip install --upgrade pip==25.3 || true

# Remove old conda metadata for pip 25.2 to fix vulnerability detection
RUN rm -f /opt/conda/envs/ptca/conda-meta/pip-25.2-pyhc872135_0.json || true

# This line is added because azureml-automl-dnn-vision depends on python 3.9.
# Added ignore python requires to get this installed.
# RUN pip install --ignore-requires-python azureml-automl-dnn-vision
RUN pip install --no-deps azureml-automl-dnn-vision

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install --upgrade nltk==3.9

# Adding this to exclude Python 3.13 packages from the setuptools library fix
RUN rm -rf /opt/conda/lib/python3.13 || true

# List pip packages
RUN pip list

## Delete
RUN rm requirements.txt