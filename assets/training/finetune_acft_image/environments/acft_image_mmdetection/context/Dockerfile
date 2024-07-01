# PTCA image
<<<<<<< HEAD
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py310-torch201:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
=======
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch201:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# # # Install mmdet
# # Note that MMDet installs pycocotools
# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==3.3.0

<<<<<<< HEAD
=======
# Vulnerability Fix
RUN pip install gunicorn==22.0.0

>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
# dummy number to change when needing to force rebuild without changing the definition: 1