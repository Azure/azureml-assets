FROM mcr.microsoft.com/azureml/inference-base-2004:{{latest-image-tag}}

RUN conda create -n minimal \
    python=3.8.13 pip=22.1.2

ENV AZUREML_CONDA_ENVIRONMENT_PATH /opt/miniconda/envs/minimal

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

RUN pip install 'azureml-inference-server-http~=0.7.0'

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

CMD [ "runsvdir", "/var/runit" ]