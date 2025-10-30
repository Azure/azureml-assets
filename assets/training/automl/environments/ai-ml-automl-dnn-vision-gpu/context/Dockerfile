FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
RUN apt-get update && \
    apt-get install -y --only-upgrade \
        libpython3.10-stdlib \
        python3.10 \
        libpython3.10-minimal \
        python3.10-minimal \
        libpam0g \
        libpam-modules-bin \
        libpam-modules \
        libpam-runtime && \
    apt-get install -y sudo=1.9.9-1ubuntu2.5 && \
    apt-mark hold sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        libc-bin \
        libssl-dev \
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

ENV ENABLE_METADATA=true

# Create conda environment
# begin conda create
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.9 \
    # begin conda dependencies
    pip=21.3.1 \
    numpy~=1.23.5 \
    libffi=3.3 \
    pycocotools=2.0.4 \
    shap=0.39.0 \
    llvmlite=0.39.1 \
    scipy=1.10.1 \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    tbb=2021.1.1 \
    # end conda dependencies
    -c conda-forge -c cerebis && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y
# end conda create

# begin pip install

# Install pip dependencies
RUN pip install \
                 # begin pypi dependencies
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-automl-dnn-vision=={{latest-pypi-version}} \
                'azureml-dataprep>=2.24.4' \
                'requests>=2.32.4' \
                'azure-identity>=1.16.1'
                # end pypi dependencies

# Update cryptography and pyarow for fixing vulnerability. Doing it  separately from pip install to avoid conflict with other packages
RUN pip install cryptography>=42.0.5 \
                pyarrow==14.0.2 \
                aiohttp>=3.12.14

# Patch for pillow vulnerability
RUN /azureml-envs/azureml-automl-dnn-vision-gpu/bin/pip install --upgrade 'pillow==11.3.0'
RUN /opt/conda/bin/pip install --upgrade 'pillow==11.3.0' 
RUN /opt/conda/envs/ptca/bin/pip install --upgrade 'pillow==11.3.0'
                
# Upgrade torch, requests, urllib3 in the system Python for fixing vulnerability
RUN /opt/conda/bin/pip install --upgrade 'urllib3==2.5.0'
RUN /opt/conda/envs/ptca/bin/pip install --upgrade 'urllib3==2.5.0'
RUN /azureml-envs/azureml-automl-dnn-vision-gpu/bin/pip install --upgrade 'urllib3==2.5.0'

RUN /azureml-envs/azureml-automl-dnn-vision-gpu/bin/pip install --force-reinstall --no-cache-dir 'torch==2.8.0' torchvision==0.23.0
RUN /opt/conda/envs/ptca/bin/pip install --force-reinstall --no-cache-dir 'torch==2.8.0' torchvision==0.23.0
RUN /azureml-envs/azureml-automl-dnn-vision-gpu/bin/pip install --upgrade 'mlflow==3.1.4'

RUN $AZUREML_CONDA_ENVIRONMENT_PATH/bin/pip install --upgrade 'h2>=4.3.0'
RUN rm -rf /opt/conda/pkgs/

# end pip install
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
