#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131:biweekly.202301.2
RUN pip uninstall -y onnxruntime-training
RUN pip install -i https://test.pypi.org/simple/ onnxruntime-training==1.14.0.dev20230119001

# ENV NCCL_DEBUG=INFO
 
# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

# pip install requirements
RUN pip install https://automlcesdkdataresources.blob.core.windows.net/finetuning-models/prp_wheels/azureml_acft_accelerator-0.1.0.0-py3-none-any.whl

# Check versions
RUN pip list

