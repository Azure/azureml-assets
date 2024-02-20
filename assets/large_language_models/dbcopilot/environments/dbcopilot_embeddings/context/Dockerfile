FROM mcr.microsoft.com/azureml/promptflow/promptflow-runtime-stable:20240205.v2

USER root
# Install necessary tools
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc-dev \
    # Upgrade libldap to the required version
    libldap-2.4-2=2.4.49+dfsg-2ubuntu1.10 \
    libldap-common=2.4.49+dfsg-2ubuntu1.10 && \
    rm -rf /var/lib/apt/lists/*

# Import the public repository GPG keys
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# Register the Microsoft Ubuntu repository
RUN curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Update the list of products
RUN apt-get update

# Install MS SQL ODBC Driver
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

## Install dbcopilot
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.15-py3-none-any.whl

RUN pip install cryptography==41.0.6 langchain==0.0.348 pyarrow==14.0.1 pymongo==4.6.1 promptflow-image-bundle[azure]==0.2.0

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