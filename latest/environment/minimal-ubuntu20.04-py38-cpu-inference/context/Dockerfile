FROM mcr.microsoft.com/azureml/inference-base-2004@sha256:25132eb4fb162d4a3c2f58f6b9ef005228b21f2227b4a0abd56f5667885558bc

WORKDIR /
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/minimal
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

# Create conda environment
USER root
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y   
USER dockeruser

CMD [ "runsvdir", "/var/runit" ]