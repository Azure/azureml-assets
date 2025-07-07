FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch260

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-forecasting-gpu
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Optional: Copy MLflow score script from a known image if needed
# COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
ENV ENABLE_METADATA=true

# Create conda environment with conda dependencies
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    pip=22.1.2 \
    numpy~=1.23.5 \
    scikit-learn=1.5.1 \
    pandas~=1.5.3 \
    scipy=1.10.1 \
    'psutil>=5.2.2,<6.0.0' \
    tqdm \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    -c conda-forge -c pytorch -c anaconda && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH python --version && \
    conda clean -a -y

# Install pip dependencies with AzureML v2 SDK
RUN pip install \
    marshmallow==3.19.0 \
    mlflow \
    azure-ai-ml==1.16.0 \
    azure-identity==1.16.1 \
    inference-schema \
    xgboost==1.5.2 \
    cryptography>=42.0.5 \
    requests==2.32.4 \
    certifi>=2023.07.22 \
    spacy==3.7.4 \
    GitPython>=3.1.41 \
    https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz \
    py-cpuinfo==5.0.0

# Install PyTorch before Horovod
RUN pip install torch==2.6.0 torchvision torchaudio

# Optional: Install Horovod for distributed training
RUN HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir \
    git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d