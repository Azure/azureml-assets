FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:{{latest-image-tag}}
WORKDIR /

USER root

RUN apt-get update \
<<<<<<< HEAD
    && apt-get install -y --no-install-recommends curl unixodbc gnupg2 python3.9 \
    && curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc \
    && curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get install -y libpam0g=1.4.0-11ubuntu2.4 \
=======
    && apt-get install -y --no-install-recommends curl unixodbc gnupg2 python3.10 \
    && curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc \
    && curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18=18.3.3.1-1 \
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
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
<<<<<<< HEAD
    pip install promptflow-image-bundle[azure]==0.2.0 && \
    ## Install dbcopilot
    pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.18-py3-none-any.whl langchain==0.0.348
=======
    pip install promptflow-image-bundle[azure]==0.3.0 && \
    pip install promptflow-vectordb==0.2.10 && \
    ## Install dbcopilot
    pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.23-py3-none-any.whl
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa

# Fix vunerabilities
RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate /opt/miniconda/envs/amlenv && \
<<<<<<< HEAD
    pip install cryptography==42.0.5 langchain==0.0.348 && \
=======
    pip install sqlparse==0.5.0 gunicorn==22.0.0 Werkzeug==3.0.3 && \
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
    conda deactivate"

# For GUNICORN_CMD_ARGS, we need to set the timeout to be 0 so that the server will not timeout
# ENV AML_CORS_ORIGINS="*"