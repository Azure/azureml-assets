FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x:biweekly.202509.2

USER root
RUN apt-get -y update

COPY requirements.txt .

RUN pip install -r requirements.txt