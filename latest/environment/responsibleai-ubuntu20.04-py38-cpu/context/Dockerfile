FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20231130.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/responsibleai

# Install wkhtmltopdf for pdf rendering from html
RUN apt-get -y update && apt-get -y install wkhtmltopdf

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

RUN pip install --pre 'azure-ai-ml' 'azure-storage-blob<=12.13.0' 'numpy<1.24.0'

# no-deps install for domonic due to unresolvable dependencies requirment on urllib3 and requests.
# score card rendering is using domonic only for the html elements composer which does not involve requests or urllib3
RUN pip install --no-deps 'charset-normalizer==2.0.12' \
                          'cssselect==1.1.0' \
                          'elementpath==2.5.0' \
                          'html5lib==1.1' \
                          'webencodings==0.5.1' \
                          'domonic==0.9.10'

# Install azureml packages
RUN pip install 'azureml-dataset-runtime==1.54.0' \
                'azureml-core==1.54.0' \
                'azureml-mlflow==1.54.0' \
                'azureml-telemetry==1.54.0'

# To resolve vulnerability issue
RUN pip install 'Werkzeug==2.2.3'

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
