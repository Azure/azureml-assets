FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:{{latest-image-tag}}
WORKDIR /

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unixodbc gnupg2 python3.9 \
    && curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc \
    && curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get install -y libpam0g=1.4.0-11ubuntu2.4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# NGINX configuration
RUN NGINX_CONF=/etc/nginx/sites-enabled/app; \
    sed -i "$(grep "location /" $NGINX_CONF -n | cut -f1 -d:) a proxy_buffering off;" $NGINX_CONF

# Update pip
RUN python -m pip install --upgrade pip

# Install necessary tools
COPY ./requirements.txt ./
RUN pip install -r requirements.txt

# Install promptflow environment
RUN pip install promptflow-image-bundle[azure]==0.2.0

## Install dbcopilot
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.17-py3-none-any.whl

# Fix vunerabilities
RUN pip install cryptography==42.0.5

# For GUNICORN_CMD_ARGS, we need to set the timeout to be 0 so that the server will not timeout
# ENV AML_CORS_ORIGINS="*"