FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20240211.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/responsibleai-text

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

RUN pip install --pre 'azure-ai-ml' 'azure-storage-blob<=12.13.0' 'numpy<1.24.0'

# Install main azureml packages
RUN pip install 'azureml-core==1.55.0' \
                'azureml-mlflow==1.55.0' \
                'azureml-rai-utils==0.0.6'

RUN pip install 'azureml-dataset-runtime==1.55.0' \
                'azureml-telemetry==1.55.0' \
                'Werkzeug==2.2.3'

# azureml-dataset-runtime[fuse] upper bound for pyarrow is 11.0.0
# so we install pyarrow in extra step to avoid conflict
RUN pip install 'pyarrow>=14.0.1'

# Install huggingface evaluate package from source
# see issue https://github.com/huggingface/datasets/issues/6182
RUN pip install git+https://github.com/huggingface/evaluate.git

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
