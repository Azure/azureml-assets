#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131@sha256:2d1a82c6c2721ac0c51adb1862b3a36d041bdd38b568b4a78cbfc60950a5cb5b

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

COPY requirements.txt .
RUN pip install "azureml-evaluate-mlflow==0.0.6.post1"
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt

# Check versions
RUN pip list
