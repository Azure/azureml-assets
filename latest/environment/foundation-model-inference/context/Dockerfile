# Use the base image that includes the necessary dependencies
FROM ghcr.io/huggingface/text-generation-inference:0.9

ENV DEBIAN_FRONTEND=noninteractive AZUREML_ENTRY_SCRIPT="mlflow_score_script.py" AML_APP_ROOT="/var/mlflow_resources" \ 
    SVDIR=/var/runit WORKER_TIMEOUT=3600

# Remove cached conda packages (which contains some vulnerable packages as well)
RUN rm -rf /opt/conda/pkgs
RUN apt-get upgrade && apt-get update -y && apt-get install git git-lfs vim openssh-server openssh-client curl graphviz -y

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y openssh-server openssh-client

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir && rm requirements.txt

# Copy scoring file
COPY score.py ${AML_APP_ROOT}/${AZUREML_ENTRY_SCRIPT}

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
EXPOSE 5001 8883 8888

# clear entrypoint
ENTRYPOINT []

CMD [ "runsvdir", "/var/runit" ]
