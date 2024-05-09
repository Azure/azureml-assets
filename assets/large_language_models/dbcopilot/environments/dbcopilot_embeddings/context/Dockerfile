FROM mcr.microsoft.com/azureml/promptflow/promptflow-runtime-stable:{{latest-image-tag}}
WORKDIR /

USER root

# Install necessary tools
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc-dev \
    openssl=1.1.1f-1ubuntu2.22 \
    libssl1.1=1.1.1f-1ubuntu2.22 \
    libldap-2.4-2=2.4.49+dfsg-2ubuntu1.10 \
    libldap-common=2.4.49+dfsg-2ubuntu1.10 \
    libxml2=2.9.10+dfsg-5ubuntu0.20.04.7 \
    login=1:4.8.1-1ubuntu5.20.04.5 \
    passwd=1:4.8.1-1ubuntu5.20.04.5 \
    uidmap=1:4.8.1-1ubuntu5.20.04.5 \
    libcurl3-gnutls=7.68.0-1ubuntu2.22 \
    libuuid1=2.34-0.1ubuntu9.6  \
    libblkid1=2.34-0.1ubuntu9.6\
    util-linux=2.34-0.1ubuntu9.6 \
    fdisk=2.34-0.1ubuntu9.6 \
    libfdisk1=2.34-0.1ubuntu9.6 \
    mount=2.34-0.1ubuntu9.6 \
    libsmartcols1=2.34-0.1ubuntu9.6 \
    libmount1=2.34-0.1ubuntu9.6 &&\
    rm -rf /var/lib/apt/lists/*

# Install MS SQL ODBC Driver
RUN apt-get update \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18=18.3.3.1-1 msodbcsql17=17.10.6.1-1 \
    && rm -rf /var/lib/apt/lists/*

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

# Set the shared environment path
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/prompt-flow/runtime/
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Copy the environment file
COPY ./requirements.txt ./

# dbcopilot environment
RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate $AZUREML_CONDA_ENVIRONMENT_PATH && \
    pip install --upgrade pip && \
    pip install  -r requirements.txt && \
    pip install  promptflow-vectordb==0.2.9 && \
    pip install  https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.20-py3-none-any.whl &&\
    pip install  cryptography==42.0.5 langchain==0.1.11 idna==3.7 sqlparse==0.5.0 gunicorn==22.0.0"

# Fix vunerabilities
RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate /opt/miniconda && \
    pip install cryptography==42.0.5 idna==3.7 sqlparse==0.5.0 gunicorn==22.0.0 && \
    conda deactivate"

# Install certificates and other utilities
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    # utilities for keeping Debian and OpenJDK CA certificates in sync
    ca-certificates p11-kit wget \
    ; \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/local/openjdk-21
ENV PATH $JAVA_HOME/bin:$PATH

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

# https://jdk.java.net/
# >
# > Java Development Kit builds, from Oracle
# >
ENV JAVA_VERSION 21-ea+22

COPY setup.sh .
RUN bash setup.sh