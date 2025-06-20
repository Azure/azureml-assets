FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20250619.v1

WORKDIR /
ENV CONDA_PREFIX=/azureml-envs/responsibleai-text
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml

# Prepend path to AzureML conda environment
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN conda list

RUN pip install --pre 'azure-ai-ml' 'azure-storage-blob<=12.13.0' 'numpy==1.26.2'

# Separating RAI install due to very old statsmodels dependency
# which conflicts with econml and numpy
RUN pip install 'responsibleai~=0.36.0' \
                'raiwidgets~=0.36.0' \
                'responsibleai-text[qa]==0.2.7'

# Install main azureml packages
RUN pip install 'azureml-mlflow==1.60.0.post1' \
                'azureml-rai-utils==0.0.6'

RUN pip install 'azureml-dataset-runtime==1.60.0' \
                'azureml-telemetry==1.60.0' \
                'gunicorn>=22.0.0'

# azureml-dataset-runtime[fuse] upper bound for pyarrow is 11.0.0
# so we install pyarrow in extra step to avoid conflict
RUN pip install 'pyarrow>=14.0.1'

# To resolve vulnerability issue regarding crytography
RUN pip install 'cryptography>=43.0.1'

# Install huggingface evaluate package from source
# see issue https://github.com/huggingface/datasets/issues/6182
RUN pip install git+https://github.com/huggingface/evaluate.git

# To resolve vulnerability issue
RUN pip install 'Werkzeug>=3.0.3'
RUN pip install 'tqdm>=4.66.3'
RUN pip install 'setuptools>=78.1.1'

# clean conda and pip caches
RUN conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

RUN rm -rf ~/.cache/pip

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $CONDA_PREFIX/lib:$LD_LIBRARY_PATH
