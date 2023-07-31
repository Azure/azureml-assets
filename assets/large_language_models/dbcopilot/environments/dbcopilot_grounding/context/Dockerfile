FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/dbcopilot
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

USER root

# ODBC
# adding custom MS repository
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# install SQL Server drivers
# install SQL Server tools
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18
RUN echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"


# Workaround: install packages with extra index url. Will be removed after the packages are published to pypi
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/dbcopilot-0.3.20230714.1-py3-none-any.whl
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.7-py3-none-any.whl

# Install promptflow
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/promptflow_sdk-0.0.99116360-py3-none-any.whl
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/prompt_flow_tools-0.0.3-py3-none-any.whl

# Install embeddingstore sdk
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/embeddingstore-0.0.99061601-py3-none-any.whl


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