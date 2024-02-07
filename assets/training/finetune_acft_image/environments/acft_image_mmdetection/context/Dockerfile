# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py310-torch201:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# # # Install mmcv-full and mmdet
# RUN pip install openmim
RUN mim install mmcv==2.0.1
# # Note that MMDet installs pycocotools
# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==3.1.0

# Vulnerability fixes
RUN pip install pydash==6.0.0
RUN pip install urllib3==2.0.7
RUN pip install pyarrow==14.0.1
RUN pip install aiohttp==3.9.1
# dummy number to change when needing to force rebuild without changing the definition: 1