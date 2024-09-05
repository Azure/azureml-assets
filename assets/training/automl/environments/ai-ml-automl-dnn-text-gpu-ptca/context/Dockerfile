FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py310-torch1131:{{latest-image-tag}}

USER root:root

RUN pip install 'azureml-automl-dnn-nlp=={{latest-pypi-version}}'
RUN pip install 'azureml-defaults=={{latest-pypi-version}}'

RUN pip install torch==1.13.1
RUN pip uninstall -y onnxruntime-training
RUN pip install onnxruntime-training==1.15.1
RUN pip install torch-ort && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python -m torch_ort.configure

RUN pip install transformers==4.36.2
RUN pip install optimum==1.16.1
RUN pip install accelerate==0.26.1
RUN pip install deepspeed==0.13.1
RUN pip install numpy==1.22.0

# Address vulnerabilities
RUN pip install pyarrow==14.0.1
RUN pip install aiohttp==3.10.2
RUN pip install idna==3.7
RUN pip install requests==2.32.3

# dummy number to change when needing to force rebuild without changing the definition: 14