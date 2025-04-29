FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20250427.v1

# Add Docker's official GPG key:
RUN apt-get update && \
    apt-get install ca-certificates curl && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc 
    
# Add the repository to Apt sources:
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && \
    apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

ENV TRIVY_VERSION=0.58.1
ENV ORAS_VERSION=1.2.2

RUN wget https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-64bit.deb && \
    dpkg -i trivy_${TRIVY_VERSION}_Linux-64bit.deb && \
    rm trivy_${TRIVY_VERSION}_Linux-64bit.deb && \
    trivy image --download-db-only && \
    trivy image --download-java-db-only
	
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
