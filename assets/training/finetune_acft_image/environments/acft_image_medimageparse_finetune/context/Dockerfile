FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update

COPY requirements.txt .

RUN pip install -r requirements.txt