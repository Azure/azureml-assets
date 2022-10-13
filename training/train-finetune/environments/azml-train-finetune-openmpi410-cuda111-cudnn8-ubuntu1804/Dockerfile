FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:20220616.v1

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

RUN pip install \
        datasets==2.3.2 \
        mpi4py==3.1.3 \
        mlflow==1.28.0 \
        accelerate==0.11.0 \
        optimum==1.3.0 \
        transformers==4.21.1

# Dependencies for Summarization, Translation
RUN pip install rouge-score==0.1.2 \
        sacrebleu==2.2.1
RUN python -m nltk.downloader punkt

# torch 11.3
RUN pip uninstall -y torch
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

# Dependecies for model deployment
RUN pip install \
        azure-ai-ml==0.1.0b5 \
        azureml-inference-server-http==0.7.2 \
        azureml-contrib-services==1.41.0 \
        MarkupSafe==2.0.1

# DS + ORT + Torch ORT
RUN pip install deepspeed

# Check versions
RUN pip list

# Check versions
RUN pip list | grep -E "gllm|pyarrow|transformers|deepspeed|azureml-contrib-services"