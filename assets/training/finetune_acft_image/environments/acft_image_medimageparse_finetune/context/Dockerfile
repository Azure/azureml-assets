FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@a1ce2f9

# protobuf is a transitive dep of mlflow-skinny/onnx; parents require >=3.12.0, cannot force 6.33.5
RUN pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 protobuf==6.33.5

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 pip==26.0 wheel==0.46.2 setuptools==82.0.0