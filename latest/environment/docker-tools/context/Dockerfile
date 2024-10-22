FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20241020.v1

RUN apt-get update
RUN apt-get install docker.io -y

ENV TRIVY_VERSION=0.53.0
ENV ORAS_VERSION=1.2.0

RUN wget https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-64bit.deb && \
    dpkg -i trivy_${TRIVY_VERSION}_Linux-64bit.deb && \
    rm trivy_${TRIVY_VERSION}_Linux-64bit.deb
	
RUN wget https://github.com/oras-project/oras/releases/download/v${ORAS_VERSION}/oras_${ORAS_VERSION}_linux_amd64.tar.gz && \
	mkdir -p oras-install/ && \
	tar -zxf oras_${ORAS_VERSION}_*.tar.gz -C oras-install/ && \
	mv oras-install/oras /usr/local/bin/ && \
	rm -rf oras_${ORAS_VERSION}_*.tar.gz oras-install/

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/image-build

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    pip=24.* -c conda-forge

# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install azure-storage-blob==12.20.0