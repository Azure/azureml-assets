# PTCA image
FROM mcr.microsoft.com/aifx/acpt/vision/stable-ubuntu2004-cu117-py38-torch1131:biweekly.202310.4

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# # # Install mmcv-full and mmdet
# RUN pip install openmim
RUN mim install mmtrack==0.14.0
RUN mim install mmcv-full==1.7.1
# # Note that MMDet installs pycocotools
# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==2.28.2

# vulnerability fixes - this will be removed once we update to MMTracking's latest version
RUN pip install yapf==0.40.1
RUN pip install pydash==6.0.0
RUN pip install urllib3==2.0.7