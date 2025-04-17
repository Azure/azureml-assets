FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch222:biweekly.202504.1.v1

WORKDIR /

RUN apt-get update && apt-get upgrade -y && apt-get install wget nscd python3-idna git git-lfs -y

# azcopy install starts
RUN echo "Downloading azcopy to file azcopy.tar ....";\
    wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar --no-verbose;\
    tar -xvf azcopy.tar;\
    mkdir -p ./azcopy/bin/;\
    cp ./azcopy_linux_amd64_*/azcopy ./bin/;\
    rm azcopy.tar;\
    rm -rf ./azcopy_linux_amd64_*;\
    which azcopy | grep -o azcopy > /dev/null &&  echo "azcopy not installed" || echo "azcopy installed";

COPY requirements.txt .

# Switch to python==3.9 to prevent numpy==1.23.5 build failure
# numpy==1.23.5 is a hard dependency in azureml-automl-dnn-vision
# which is not compatible with py310.
# Going forward  azureml-automl-dnn-vision needs to update version for torch and numpy
RUN conda install python=3.9 -y

RUN pip install -r requirements.txt --no-cache-dir

# Vulnerability fix
RUN pip install Pillow gunicorn onnx==1.17.0 idna tqdm requests==2.32.1 tornado==6.4.2 certifi==2024.07.04 urllib3==1.26.19 scikit-learn==1.5.1 mlflow==2.19 mlflow-skinny==2.19 marshmallow==3.23.2

# List pip packages
RUN pip list

## Delete
RUN rm requirements.txt
