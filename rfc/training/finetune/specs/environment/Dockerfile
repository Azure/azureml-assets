#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu113-py38-torch1110:latest
 
# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

# pip install requirements
RUN pip install https://automlcesdkdataresources.blob.core.windows.net/finetuning-models/azureml_gllm_new-0.0.2-py3-none-any.whl
RUN pip install \
        datasets==2.3.2 \
        mpi4py==3.1.3 \
        # mlflow==1.27.0 \
        accelerate==0.11.0 \
        optimum==1.3.0 \
        transformers==4.21.1

# Dependecies for model deployment
RUN pip install \
        azure-ai-ml==0.1.0b5 \
        azureml-inference-server-http==0.7.2 \
        azureml-contrib-services==1.43.0 \
        MarkupSafe==2.0.1

# Check versions
RUN pip list
