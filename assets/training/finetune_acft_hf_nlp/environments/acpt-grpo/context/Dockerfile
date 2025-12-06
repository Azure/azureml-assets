#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
