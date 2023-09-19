FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202309.2

WORKDIR /

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y openssh-server openssh-client

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# List installed packages
RUN pip list

## Delete
RUN rm requirements.txt

# Copy scoring file
COPY score.py /var/mlflow_resources/mlflow_score_script.py
ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"
ENV AML_APP_ROOT="/var/mlflow_resources"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001 8883 8888

CMD [ "runsvdir", "/var/runit" ]