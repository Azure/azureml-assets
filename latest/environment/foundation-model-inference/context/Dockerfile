FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch201:biweekly.202311.1

WORKDIR /

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update && apt-get -y upgrade

COPY . .
# upgrade torch
RUN pip install --no-cache-dir -e .
RUN pip install --upgrade --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1
RUN pip list    # show installed packages

# Copy scoring file
COPY src/score.py /var/mlflow_resources/mlflow_score_script.py
ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"
ENV AML_APP_ROOT="/var/mlflow_resources"

# clean conda and pip caches
RUN conda clean -y --all
RUN rm -rf ~/.cache/pip

#Inference requirements
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
