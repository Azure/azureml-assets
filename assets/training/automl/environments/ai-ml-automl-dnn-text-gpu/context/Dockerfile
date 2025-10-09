FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch260:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-text-gpu
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

# begin conda create
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    # begin conda dependencies
    pip=22.1.2 \
    numpy~=1.23.5 \
    scikit-learn=1.5.1 \
    pandas~=1.5.3 \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    scipy=1.10.1 \
    pybind11=2.10.1 \
    # end conda dependencies
    -c conda-forge -c anaconda

# Ensure additional conda and pip install commands apply to our conda env of interest.
SHELL ["conda", "run", "-p", "$AZUREML_CONDA_ENVIRONMENT_PATH", "/bin/bash", "-c"]

# begin pip install
# Install pip dependencies
# Here, we pin sentencepiece since 0.1.98 breaks training. Earlier versions of horovod contain a sev2 vulnerability,
# and earlier versions of tokenizers cause log spam with transformers==4.16.0.
RUN pip install \
                # begin pypi dependencies
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-dnn-nlp=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                'azure-identity>=1.16.1' \
                'sentencepiece==0.1.97' \
                'cryptography>=42.0.5'\
                'requests>=2.32.4' \
                'certifi>=2023.07.22' 
                # end pypi dependencies

# Horovod incompatible with torch>2.0 https://github.com/horovod/horovod/issues/3996
RUN HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d

# Separate updates for fixing vulnerabilities.
# Doing it  separately from pip install above to avoid conflict with other packages
# We should aim for this list to be empty with new and patched releases
# by fixing dependencies in the base packages
RUN pip list && \
    pip install pyarrow==14.0.2 \
                'transformers[sentencepiece,torch]==4.53.0' \
                aiohttp==3.12.14
# Security vulnerability fix - GHSA-887c-mr87-cxwp: Upgrade torch to 2.8.0
RUN pip install --upgrade pip && pip install torch==2.8.0 torchvision==0.22.1
RUN /opt/conda/envs/ptca/bin/pip install --upgrade pip && /opt/conda/envs/ptca/bin/pip install torch==2.8.0 torchvision==0.22.1 || true
RUN /azureml-envs/azureml-automl-dnn-text-gpu/bin/pip install --upgrade 'urllib3==2.5.0' || true
RUN /opt/conda/bin/pip install --upgrade 'urllib3==2.5.0' || true
RUN /opt/conda/envs/ptca/bin/pip install --upgrade 'urllib3==2.5.0' || true

# Patch for pillow vulnerability
RUN /azureml-envs/azureml-automl-dnn-text-gpu/bin/pip install --upgrade 'pillow==11.3.0' || true
RUN /opt/conda/bin/pip install --upgrade 'pillow==11.3.0' || true
RUN /opt/conda/envs/ptca/bin/pip install --upgrade 'pillow==11.3.0' || true
RUN $AZUREML_CONDA_ENVIRONMENT_PATH/bin/pip install --upgrade 'h2>=4.3.0'
RUN rm -rf /opt/conda/pkgs/
# end pip install
