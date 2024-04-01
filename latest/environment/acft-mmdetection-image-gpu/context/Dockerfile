# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py310-torch201:biweekly.202404.1

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# # # Install mmdet
# # Note that MMDet installs pycocotools
# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==3.3.0

# dummy number to change when needing to force rebuild without changing the definition: 1