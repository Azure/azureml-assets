FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu113-py38-torch1110:{{latest-image-tag:monthly\.\d{6}}}

# Install pip dependencies
RUN pip install 'ipykernel~=6.0' \
                '{pin-to-latest-pypi-version:"azureml-core=={}"}' \
				'{pin-to-latest-pypi-version:"azureml-dataset-runtime=={}"}' \
                '{pin-to-latest-pypi-version:"azureml-defaults=={}"}' \
				'{pin-to-latest-pypi-version:"azure-ml=={}"}' \
				'{pin-to-latest-pypi-version:"azure-ml-component=={}"}' \
                '{pin-to-latest-pypi-version:"azureml-mlflow=={}"}' \
                '{pin-to-latest-pypi-version:"azureml-telemetry=={}"}' \
                'torch-tb-profiler~=0.4.0' \
				'py-spy==0.3.12'

RUN pip install \
        azure-ai-ml==0.1.0b5 \
        azureml-inference-server-http~=0.7.0 \
        inference-schema~=1.4.2.1 \
        azureml-contrib-services~=1.41.0 \
        MarkupSafe==2.0.1

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20220607.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

