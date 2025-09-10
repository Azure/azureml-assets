FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update

COPY requirements.txt .
RUN pip install azureml_acft_common_components
RUN pip install azureml_acft_image_components[mip] --no-dependencies
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'