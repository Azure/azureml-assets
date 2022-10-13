#PTCA image
#FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu113-py38-torch1110
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-py38-cuda11.3-gpu

# ENV NCCL_DEBUG=INFO

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

# ENV DISABLE_MLFLOW_INTEGRATION=TRUE
ENV AZML_DEPLOYMENT_ENV_NAME=azml-train-finetune-deploy-curated-acpt-pytorch-111-py38-cuda113-gpu
ENV AZML_REGISTRY_NAME=

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
        azureml-inference-server-http~=0.7.0 \
        inference-schema~=1.4.2.1 \
        azureml-contrib-services~=1.41.0 \
        MarkupSafe==2.0.1

# Check versions
RUN pip list

# Dependencies for online deployment
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20220607.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh
RUN cp /var/configuration/rsyslog.conf /etc/rsyslog.conf
RUN cp /var/configuration/nginx.conf /etc/nginx/sites-available/app
RUN ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app
RUN rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888
