    FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:{{latest-image-tag}}

    ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-forecasting-gpu
    # Prepend path to AzureML conda environment
    ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

    COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

    ENV MLFLOW_MODEL_FOLDER="mlflow-model"
    # ENV AML_APP_ROOT="/var/mlflow_resources"
    # ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

    ENV ENABLE_METADATA=true

    RUN apt-get update && \
    apt-get install -y --only-upgrade \
        systemd \
        libsystemd0 \
        libnss-systemd \
        libpam-systemd \
        libudev1 \
        systemd-sysv \
        systemd-timesyncd \
        python3.10 \
        python3.10-minimal \
        libarchive13 \
        libpython3.10-minimal \
        libpython3.10-stdlib \
        libpam0g \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


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
        pytorch=2.7.1 \
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
                    azureml-core=={{latest-pypi-version}} \
                    azureml-mlflow=={{latest-pypi-version}} \
                    azureml-defaults=={{latest-pypi-version}} \
                    azureml-telemetry=={{latest-pypi-version}} \
                    azureml-interpret=={{latest-pypi-version}} \
                    azureml-responsibleai=={{latest-pypi-version}} \
                    azureml-automl-core=={{latest-pypi-version}} \
                    azureml-automl-runtime=={{latest-pypi-version}} \
                    azureml-train-automl-client=={{latest-pypi-version}} \
                    azureml-train-automl-runtime=={{latest-pypi-version}} \
                    azureml-dataset-runtime=={{latest-pypi-version}} \
                    azureml-train-automl=={{latest-pypi-version}} \
                    azureml-contrib-automl-dnn-forecasting=={{latest-pypi-version}} \
                    'azure-identity>=1.16.1' \
                    'inference-schema' \
                    'xgboost==1.5.2' \
                    'cryptography>=42.0.5' \
                    'requests>=2.32.4' \
                    'certifi>=2023.07.22' \
                    'spacy==3.7.4' \
                    'GitPython>=3.1.41' \
                    'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz' \
                    'py-cpuinfo==5.0.0' \
                    "pillow>=11.3.0"
                    # end pypi dependencies

    RUN HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d
    # end pip install

    # Fix vulnerabilities
    RUN pip install --upgrade urllib3
    RUN pip install torch==2.7.1 torchvision==0.22.1