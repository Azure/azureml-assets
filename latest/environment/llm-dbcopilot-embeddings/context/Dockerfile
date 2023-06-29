FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20230628.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/dbcopilot
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

USER root

# ODBC
# adding custom MS repository
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list

# install SQL Server drivers
# install SQL Server tools
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18
RUN echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"


# Workaround: install packages with extra index url. Will be removed after the packages are published to pypi
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/dbcopilot-0.3.20230613.dev4-py3-none-any.whl
RUN pip install https://ragsample.blob.core.windows.net/ragdata/wheels/dbcopilot/db_copilot_tool-0.1.3-py3-none-any.whl
