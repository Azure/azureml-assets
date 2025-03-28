# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
FROM mcr.microsoft.com/azureml/o16n-base/python-assets:20250123.v1 AS inferencing-assets

FROM mcr.microsoft.com/azurelinux/base/core:3.0

USER root:root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN tdnf update && \
    tdnf install -y --no-install-recommends \
        procps \ 
        libgnutls30 \ 
        libk5crypto3 \ 
        libkrb5-3 \ 
        libkrb5support0 \ 
        libpam-modules \ 
        nginx-light \
        wget \
        runit \
        libtinfo6 \ 
        libncurses6 \ 
        ncurses-bin \ 
        ncurses-base \ 
        libncursesw6 \ 
        libcap2 \ 
        libc6 \    
        libc-bin \
        libtasn1-6 \
    tdnf clean -y && \
    rm -rf /var/lib/apt/lists/*

# Inference
# Copy logging utilities, nginx and rsyslog configuration files, IOT server binary, etc.

COPY --from=inferencing-assets /artifacts /var/
RUN sed -i '/liblttng-ust0/d' /var/requirements/system_requirements.txt
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default


ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=300
ENV AZUREML_INFERENCE_SERVER_HTTP_ENABLED="True"
EXPOSE 5001

# Stores image version information and log it while running inferencing server for better Debuggability
RUN echo "BASE IMAGE NAME" >> /IMAGE_INFORMATION && \
    echo $BASE_IMAGE_NAME "\\n====================================================" >> /IMAGE_INFORMATION && \
    echo "\\nBASE_OS" >> /IMAGE_INFORMATION && \
    echo $BASE_OS "\\n=====================================================" >> /IMAGE_INFORMATION

USER dockeruser

# Conda Environment
ENV PATH=/opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_24.5.0-0-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm -f /tmp/miniconda.sh && \
    find /opt/miniconda -type d -name __pycache__ | xargs rm -rf

ENV AZUREML_CONDA_ENVIRONMENT_PATH="/opt/miniconda/envs/amlenv"
ENV PATH="/opt/miniconda/envs/amlenv/bin:$PATH"
