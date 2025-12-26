# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get install -y libglib2.0-0 libc-bin libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
RUN apt-get install -y expat

# # # Install mmdet
# # Note that MMDet installs pycocotools
# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==3.3.0
# Temporary workaround for https://github.com/open-mmlab/mmdetection/issues/11668 (when mmdet updated, remove lines below)
RUN mim install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html --no-cache-dir
RUN sed -i 's/2.2.0/2.3.0/' /opt/conda/envs/ptca/lib/python3.10/site-packages/mmdet/__init__.py

# vulnerabilities
RUN /opt/conda/bin/python -m pip install --upgrade --no-cache-dir "urllib3>=2.6.0"
