FROM mcr.microsoft.com/azureml/inference-base-gpu-2004@sha256:5c79a6593d85a386cd316948453de1689d1d67fbe7c4cc5b9129a58d61e8983a

WORKDIR /

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/minimal
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

# Create conda environment
USER root
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y   
USER dockeruser

CMD [ "runsvdir", "/var/runit" ]
