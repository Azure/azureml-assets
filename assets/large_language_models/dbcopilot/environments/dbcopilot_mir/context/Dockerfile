FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:latest
WORKDIR /

USER root

RUN NGINX_CONF=/etc/nginx/sites-enabled/app; sed -i "$(grep "location /" $NGINX_CONF -n | cut -f1 -d:) a proxy_buffering off;" $NGINX_CONF;

# Install vim
RUN apt-get update && apt-get clean && apt install -y curl && apt-get install -y unixodbc
# Install ODBC
# adding custom MS repository
RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc

RUN curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list

# install SQL Server drivers
# install SQL Server tools
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql18

# remove because promptflow need root
# USER dockeruser
RUN python -m pip install --upgrade pip
RUN pip install azureml-rag azureml-contrib-services azure-cosmos

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./extra_requirements.txt ./
RUN pip install -r extra_requirements.txt

## Install dbcopilot
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.12-py3-none-any.whl

RUN pip install cryptography==41.0.4 langchain==0.0.317 pyarrow==14.0.1

# RUN pip install plotly~=5.13.1 azure-kusto-data==4.2.0 recognizers-text-suite~=1.0.2a2

# For GUNICORN_CMD_ARGS, we need to set the timeout to be 0 so that the server will not timeout
# ENV AML_CORS_ORIGINS="*"