FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

# Create conda environment
# begin conda create
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 \
    # begin conda dependencies
    pip=21.3.1 \
    numpy~=1.22.3 \
    libffi=3.3 \
    pycocotools=2.0.4 \
    shap=0.39.0 \
    recordclass=0.14.3 \
    llvmlite=0.36.0 \
    setuptools=65.5.1 \
    wheel=0.38.1 \
    # end conda dependencies
    -c conda-forge -c cerebis && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y
# end conda create

# begin pip install

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Update cryptography for fixing vulnerability. Doing it  separately from pip install to avoid conflict with other packages
RUN pip install cryptography==41.0.0

# Install pip dependencies
RUN pip install "azureml-mlflow=={{latest-pypi-version}}" \
                "azureml-dataset-runtime=={{latest-pypi-version}}" \
                "azureml-telemetry=={{latest-pypi-version}}" \
                "azureml-responsibleai=={{latest-pypi-version}}" \
                "azureml-automl-core=={{latest-pypi-version}}" \
                "azureml-automl-runtime=={{latest-pypi-version}}" \
                "azureml-train-automl-client=={{latest-pypi-version}}" \
                "azureml-defaults=={{latest-pypi-version}}" \
                "azureml-interpret=={{latest-pypi-version}}" \
                "azureml-train-automl-runtime=={{latest-pypi-version}}" \
                "azureml-automl-dnn-vision=={{latest-pypi-version}}" \
                'azureml-dataprep>=2.24.4'
                # end pypi dependencies
# end pip install
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
