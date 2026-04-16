FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202603.1

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir "azureml-mlflow==1.56.0"
# Register scalapb descriptor at Python startup so azureml-mlflow protos load correctly
RUN /opt/conda/envs/ptca/bin/python -c \
    "import site, os; open(os.path.join(site.getsitepackages()[0], 'scalapb_preload.pth'), 'w').write('import mlflow.protos.scalapb.scalapb_pb2\n')"
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@a1ce2f9