FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/batch-score-llm
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

RUN apt-get update && apt-get -y install curl libc6-dev libc-bin

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# Patch Azure Identity
RUN wget -q https://www.nuget.org/api/v2/package/Azure.Identity/1.10.4 -O /azure.identity.1.10.4.nupkg && \
   mkdir azure.identity.1.10.4 && \
   unzip azure.identity.1.10.4.nupkg -d azure.identity.1.10.4 && \
   mv -f azure.identity.1.10.4/lib/netstandard2.0/Azure.Identity.dll azureml-envs/batch-score-llm/lib/python3.8/site-packages/azureml/dataprep/api/engineapi/bin/

# Patch Microsoft.Rest.ClientRuntime
RUN wget -q https://www.nuget.org/api/v2/package/Microsoft.Rest.ClientRuntime/2.3.24 -O /Microsoft.Rest.ClientRuntime.2.3.24.nupkg && \
   mkdir Microsoft.Rest.ClientRuntime.2.3.24 && \
   unzip Microsoft.Rest.ClientRuntime.2.3.24.nupkg -d Microsoft.Rest.ClientRuntime.2.3.24 && \
   mv -f Microsoft.Rest.ClientRuntime.2.3.24/lib/netstandard2.0/Microsoft.Rest.ClientRuntime.dll azureml-envs/batch-score-llm/lib/python3.8/site-packages/azureml/dataprep/api/engineapi/bin/
