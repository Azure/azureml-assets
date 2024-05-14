FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:{{latest-image-tag}}
WORKDIR /

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unixodbc gnupg2 python3.10 \
    && curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc \
    && curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18=18.3.3.1-1 \
    && apt-get install -y libpam0g=1.4.0-11ubuntu2.4 \
    && apt-get install -y libexpat1=2.4.7-1ubuntu0.3 \
    && apt-get install -y bash=5.1-6ubuntu1.1 \
    && apt-get install -y libuuid1=2.37.2-4ubuntu3.4 \
    && apt-get install -y libblkid1=2.37.2-4ubuntu3.4\
    && apt-get install -y util-linux=2.37.2-4ubuntu3.4 \
    && apt-get install -y mount=2.37.2-4ubuntu3.4 \
    && apt-get install -y libsmartcols1=2.37.2-4ubuntu3.4 \
    && apt-get install -y libmount1=2.37.2-4ubuntu3.4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# NGINX configuration
RUN NGINX_CONF=/etc/nginx/sites-enabled/app; \
    sed -i "$(grep "location /" $NGINX_CONF -n | cut -f1 -d:) a proxy_buffering off;" $NGINX_CONF

# Copy the environment file
COPY ./requirements.txt ./

# dbcopilot environment
RUN python -m pip install --upgrade pip && \
    # Install necessary tools
    pip install -r requirements.txt && \
    # Install promptflow environment
    pip install promptflow-image-bundle[azure]==0.3.0 && \
    pip install promptflow-vectordb==0.2.9 && \
    ## Install dbcopilot
    pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.20-py3-none-any.whl && \
    ## fix vulnerabilities
    pip install idna==3.7

# Fix vunerabilities
RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate /opt/miniconda/envs/amlenv && \
    pip install cryptography==42.0.5 langchain==0.1.11 idna==3.7 sqlparse==0.5.0 gunicorn==22.0.0 && \
    conda deactivate"

# For GUNICORN_CMD_ARGS, we need to set the timeout to be 0 so that the server will not timeout
# ENV AML_CORS_ORIGINS="*"