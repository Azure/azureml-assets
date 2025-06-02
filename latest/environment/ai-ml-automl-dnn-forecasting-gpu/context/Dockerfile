FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:20250601.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-forecasting-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

# begin conda create
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.9 \
    # begin conda dependencies
    pip=22.1.2 \
    numpy~=1.23.5 \
    scikit-learn=1.5.1 \
    pandas~=1.5.3 \
    scipy=1.10.1 \
    'psutil>=5.2.2,<6.0.0' \
    tqdm \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    # Install pytorch separately to speed up image build
    -c conda-forge -c pytorch -c anaconda && \
    conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    pytorch=2.6.0 \
    -c pytorch -c nvidia -c conda-forge -y && \
    # end conda dependencies
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y
# end conda create

# begin pip install
# Install pip dependencies
# GitPython>=3.1.41 is required for https://github.com/advisories/GHSA-2mqj-m65w-jghx and is not available in conda
RUN pip install \
                # begin pypi dependencies
                azureml-core==1.60.0.post1 \
                azureml-mlflow==1.60.0.post1 \
                azureml-defaults==1.60.0 \
                azureml-telemetry==1.60.0 \
                azureml-interpret==1.60.0 \
                azureml-responsibleai==1.60.0 \
                azureml-automl-core==1.60.0 \
                azureml-automl-runtime==1.60.0 \
                azureml-train-automl-client==1.60.0 \
                azureml-train-automl-runtime==1.60.0 \
                azureml-dataset-runtime==1.60.0 \
                azureml-train-automl==1.60.0 \
                azureml-contrib-automl-dnn-forecasting==1.60.0 \
                'azure-identity>=1.16.1' \
                'inference-schema' \
                'xgboost==1.5.2' \
                'cryptography>=42.0.5' \
                'requests>=2.31.0' \
                'certifi>=2023.07.22' \
                'spacy==3.7.4' \
                'GitPython>=3.1.41' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz' \
                'py-cpuinfo==5.0.0'
                # end pypi dependencies

RUN HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d
# end pip install
