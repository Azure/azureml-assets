# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch222:biweekly.202408.1

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# # # Install mmdet
# # Note that MMDet installs pycocotools
# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==3.3.0

# Vulnerability Fix
RUN pip install gunicorn==22.0.0

# dummy number to change when needing to force rebuild without changing the definition: 1