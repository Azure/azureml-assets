#PTCA image
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-py38-cuda11.3-gpu

# ENV NCCL_DEBUG=INFO
 
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

# Dependecies for model deployment
RUN pip install \
        azure-ai-ml==0.1.0b5 \
        azureml-inference-server-http==0.7.2 \
        azureml-contrib-services==1.43.0 \
        MarkupSafe==2.0.1

# Check versions
RUN pip list
