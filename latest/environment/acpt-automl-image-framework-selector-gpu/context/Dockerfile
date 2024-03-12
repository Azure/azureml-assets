# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch220:biweekly.202403.1.v1

RUN apt-get -y update

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# dummy number to change when needing to force rebuild without changing the definition: 1