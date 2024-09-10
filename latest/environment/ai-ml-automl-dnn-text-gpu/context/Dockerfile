FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04:20240908.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-text-gpu
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
    pandas~=1.3.5 \
    setuptools=72.1.0 \
    wheel=0.44.0 \
    scipy=1.10.1 \
    pybind11=2.10.1 \
    # end conda dependencies
    -c conda-forge -c anaconda

# Ensure additional conda and pip install commands apply to our conda env of interest.
SHELL ["conda", "run", "-p", "$AZUREML_CONDA_ENVIRONMENT_PATH", "/bin/bash", "-c"]

# Conda installs from extra channels. Separate these to avoid solver OOMs.
RUN conda install pytorch=1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia && conda clean -a -y
# end conda create


# begin pip install
# Install pip dependencies
# Here, we pin sentencepiece since 0.1.98 breaks training. Earlier versions of horovod contain a sev2 vulnerability,
# and earlier versions of tokenizers cause log spam with transformers==4.16.0.
RUN pip install \
                # begin pypi dependencies
                azureml-core==1.57.0 \
                azureml-mlflow==1.57.0.post1 \
                azureml-automl-core==1.57.0 \
                azureml-automl-dnn-nlp==1.57.0 \
                azureml-automl-runtime==1.57.0 \
                azureml-train-automl-client==1.57.0 \
                azureml-train-automl-runtime==1.57.0 \
                azureml-defaults==1.57.0 \
                'azure-identity>=1.16.1' \
                'horovod==0.28.1' \
                'sentencepiece==0.1.97' \
                'cryptography>=42.0.5'\
                'urllib3>=1.26.18' \
                'requests>=2.31.0' \
                'certifi>=2023.07.22' 
                # end pypi dependencies

# Separate updates for fixing vulnerabilities.
# Doing it  separately from pip install above to avoid conflict with other packages
# We should aim for this list to be empty with new and patched releases
# by fixing dependencies in the base packages
RUN pip list && \
    pip install pyarrow==14.0.2 \
                'transformers[sentencepiece,torch]==4.36.2' \
                aiohttp==3.10.2

# end pip install

# Force new version - 1
