FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV ENABLE_METADATA=false

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 \
    pip=21.3.1 \
    numpy~=1.21.6 \
    libffi=3.3 \
    pycocotools=2.0.4 \
    shap=0.39.0 \
    recordclass=0.14.3 \
    llvmlite=0.36.0 \
    -c conda-forge -c cerebis && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install pip dependencies
RUN pip install azureml-core=={{latest-image-tag}}
                azureml-mlflow=={{latest-image-tag}}
                azureml-dataset-runtime=={{latest-image-tag}}
                azureml-telemetry=={{latest-image-tag}}
                azureml-responsibleai=={{latest-image-tag}}
                azureml-automl-core=={{latest-image-tag}}
                azureml-automl-runtime=={{latest-image-tag}}
                azureml-train-automl-client=={{latest-image-tag}}
                azureml-defaults=={{latest-image-tag}}
                azureml-interpret=={{latest-image-tag}}
                azureml-train-automl-runtime=={{latest-image-tag}}
                azureml-automl-dnn-vision=={{latest-image-tag}}
                'azureml-dataprep>=2.24.4'

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
