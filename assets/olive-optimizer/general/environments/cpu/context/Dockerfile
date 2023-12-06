FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:{{latest-image-tag}}

ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y libsm6 libxext6
RUN apt-get install --only-upgrade -y python3-urllib3 procps libprocps8 libk5crypto3 libgssapi-krb5-2 libkrb5-3 libkrb5support0

# Build python3.9 as the default python
RUN apt-get install -y --no-install-recommends \
        software-properties-common gnupg\
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends \
        python3.9-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    python3.9 -m venv /venv

ENV PATH=/venv/bin:$PATH

RUN pip install --upgrade pip && pip install --upgrade setuptools pyarrow
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir