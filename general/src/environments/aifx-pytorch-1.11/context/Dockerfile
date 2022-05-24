FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu115-py38-torch1110:{{latest-image-tag}}

# Install pip dependencies
RUN pip install --no-cache-dir \
    'ipykernel~=6.0' \
    'azureml-core=={{latest-pypi-version}}' \
    'azureml-defaults=={{latest-pypi-version}}' \
    'azureml-mlflow=={{latest-pypi-version}}' \
    'azureml-telemetry=={{latest-pypi-version}}' \
    'torch-tb-profiler~=0.4.0'
