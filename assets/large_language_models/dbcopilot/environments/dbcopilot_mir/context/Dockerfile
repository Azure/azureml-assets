FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:{{latest-image-tag}}
WORKDIR /

# Set noninteractive installation to avoid getting prompted during the build
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Microsoft ODBC Driver for SQL Server, and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unixodbc gnupg2 python3.10 && \
    curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18=18.3.3.1-1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Update pip and install Python packages
    python -m pip install --upgrade pip && \
    pip install promptflow-image-bundle[azure]==0.3.0 promptflow-vectordb==0.2.10 db-copilot-tool==0.0.6 azure-identity==1.17.1

# Copy the environment file
COPY ./requirements.txt ./
RUN pip install -r requirements.txt

# Update vulnerable packages
RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate /opt/miniconda/envs/amlenv && \
    pip install sqlparse==0.5.0 gunicorn==22.0.0 Werkzeug==3.0.3 certifi==2024.07.04 && \
    conda deactivate"

# Configure NGINX
RUN NGINX_CONF=/etc/nginx/sites-enabled/app && \
    sed -i "$(grep "location /" $NGINX_CONF -n | cut -f1 -d:) a proxy_buffering off;" $NGINX_CONF

# Set user to root
USER root

# Optional: ENV AML_CORS_ORIGINS="*"
