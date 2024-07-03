FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:{{latest-image-tag}}
WORKDIR /

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unixodbc gnupg2 python3.10 \
    && curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc \
    && curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18=18.3.3.1-1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# NGINX configuration
RUN NGINX_CONF=/etc/nginx/sites-enabled/app; \
    sed -i "$(grep "location /" $NGINX_CONF -n | cut -f1 -d:) a proxy_buffering off;" $NGINX_CONF

# Copy the environment file
COPY ./requirements.txt ./
RUN curl -o lemma.map.txt https://dbcopilotstorageaccount.blob.core.windows.net/dbcopilotdata/grounding/db_copilot/resource/lemma.map.txt?sv=2022-11-02&ss=bfqt&srt=sc&sp=rwdlacupiytfx&se=2024-07-03T10:59:54Z&st=2024-07-03T02:59:54Z&spr=https&sig=An%2FZJfdAMQLNiqP51lDJM1sADhBV%2BOqn%2B1VfsDOZaz4%3D
ENV DBCOPILOT_GROUNDING_LEMMA_PATH=/lemma.map.txt

# dbcopilot environment
RUN python -m pip install --upgrade pip && \
    # Install necessary tools
    pip install -r requirements.txt && \
    # Install promptflow environment
    pip install promptflow-image-bundle[azure]==0.3.0 && \
    pip install promptflow-vectordb==0.2.10 && \
    ## Install dbcopilot
    pip install https://dbcopilotstorageaccount.blob.core.windows.net/dbcopilotdata/dbcopilot/db_copilot_tool-0.1.23-py3-none-any.whl?sv=2022-11-02&ss=bfqt&srt=sc&sp=rwdlacupiytfx&se=2024-07-03T10:59:54Z&st=2024-07-03T02:59:54Z&spr=https&sig=An%2FZJfdAMQLNiqP51lDJM1sADhBV%2BOqn%2B1VfsDOZaz4%3D

# Fix vunerabilities
RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate /opt/miniconda/envs/amlenv && \
    pip install sqlparse==0.5.0 gunicorn==22.0.0 Werkzeug==3.0.3 azure-identity==1.16.1&& \
    conda deactivate"

# For GUNICORN_CMD_ARGS, we need to set the timeout to be 0 so that the server will not timeout
# ENV AML_CORS_ORIGINS="*"