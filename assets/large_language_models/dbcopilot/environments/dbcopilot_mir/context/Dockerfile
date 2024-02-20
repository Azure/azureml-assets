FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:latest
WORKDIR /

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unixodbc gnupg2 \
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

# remove because promptflow need root
# USER dockeruser
RUN python -m pip install --upgrade pip

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./extra_requirements.txt ./
RUN pip install -r extra_requirements.txt

## Install dbcopilot
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.15-py3-none-any.whl

RUN pip install cryptography==41.0.6 langchain==0.0.348 pyarrow==14.0.1 pymongo==4.6.1 promptflow-image-bundle[azure]==0.2.0

# RUN pip install plotly~=5.13.1 azure-kusto-data==4.2.0 recognizers-text-suite~=1.0.2a2

# For GUNICORN_CMD_ARGS, we need to set the timeout to be 0 so that the server will not timeout
# ENV AML_CORS_ORIGINS="*"