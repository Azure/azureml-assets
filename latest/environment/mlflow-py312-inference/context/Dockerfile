FROM mcr.microsoft.com/azureml/inference-base-2204:20260119.v1

WORKDIR /
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/mlflow
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

# Set MLFlow environment variables
ENV AML_APP_ROOT="/var/mlflow_resources"
ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

USER root

# We'll copy the HF scripts as well to enable better handling for v2 packaging. This will not require changes to the
# packages installed in the image, as the expectation is that these will all be brought along with the model.
COPY mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py
COPY mlflow_hf_score_cpu.py /var/mlflow_resources/mlflow_hf_score_cpu.py
COPY mlflow_hf_score_gpu.py /var/mlflow_resources/mlflow_hf_score_gpu.py

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y   
USER dockeruser

CMD [ "runsvdir", "/var/runit" ]