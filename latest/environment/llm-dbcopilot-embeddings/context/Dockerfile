FROM mcr.microsoft.com/azureml/promptflow/promptflow-runtime-stable:20240228.v3

USER root
# Set the shared environment path
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/dbcopilot
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda clean -a -y

# Install necessary tools
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc-dev \
    # Upgrade libldap to the required version
    libldap-2.4-2=2.4.49+dfsg-2ubuntu1.10 \
    libldap-common=2.4.49+dfsg-2ubuntu1.10 && \
    apt-get install --only-upgrade openssl=1.1.1f-1ubuntu2.21 libssl1.1=1.1.1f-1ubuntu2.21 && \
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

RUN python -m pip install --upgrade pip

COPY ./requirements.txt ./
RUN pip install opentelemetry-instrumentation-django==0.42b0 promptflow-image-bundle[azure]==0.2.0 && \
    pip install -r requirements.txt

## Install dbcopilot
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.16-py3-none-any.whl

RUN pip install cryptography==42.0.0 langchain==0.0.348 pyarrow==14.0.1 pymongo==4.6.1

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