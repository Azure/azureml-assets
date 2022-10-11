#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu113-py38-torch1110

# ENV NCCL_DEBUG=INFO
 
# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

# ENV DISABLE_MLFLOW_INTEGRATION=TRUE
ENV GLLM_DEPLOYMENT_ENV_NAME=gllm-curated-acpt-pytorch-111-py38-cuda113-gpu
ENV GLLM_REGISTRY_NAME=

# pip install requirements
ADD wheels ./wheels
RUN pip install ./wheels/*

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

# Dependecies for model deployment
RUN pip install \
        azure-ai-ml==0.1.0b5 \
        azureml-inference-server-http==0.7.2 \
        azureml-contrib-services==1.43.0 \
        MarkupSafe==2.0.1

# Check versions
RUN pip list
