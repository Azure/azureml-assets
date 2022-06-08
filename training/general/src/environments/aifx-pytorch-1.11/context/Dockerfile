FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu115-py38-torch1110:{{latest-image-tag}}

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --progress-bar off && \
    rm requirements.txt
