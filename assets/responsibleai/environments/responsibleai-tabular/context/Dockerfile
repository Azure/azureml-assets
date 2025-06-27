FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:{{latest-image-tag}}

# Install wkhtmltopdf for pdf rendering from html
RUN apt-get -y update && apt-get -y install wkhtmltopdf

WORKDIR /
ENV CONDA_PREFIX=/azureml-envs/responsibleai-tabular
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml

# Prepend path to AzureML conda environment
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN conda list

RUN pip install --pre 'azure-ai-ml>=1.26.3' 'azure-storage-blob<=12.13.0' 'numpy<1.24.0'

# no-deps install for domonic due to unresolvable dependencies requirment on urllib3 and requests.
# score card rendering is using domonic only for the html elements composer which does not involve requests or urllib3
RUN pip install --no-deps 'charset-normalizer==2.0.12' \
                          'cssselect==1.1.0' \
                          'elementpath==2.5.0' \
                          'html5lib==1.1' \
                          'webencodings==0.5.1' \
                          'domonic==0.9.10'

# Install azureml packages
RUN pip install 'azureml-dataset-runtime=={{latest-pypi-version}}' \
                'azureml-core=={{latest-pypi-version}}' \
                'azureml-mlflow=={{latest-pypi-version}}' \
                'azureml-telemetry=={{latest-pypi-version}}' \
                'azureml-rai-utils==0.0.6'

# azureml-dataset-runtime[fuse] upper bound for pyarrow is 11.0.0
# so we install pyarrow in extra step to avoid conflict
RUN pip install 'pyarrow>=14.0.1'

# To resolve vulnerability issue regarding crytography
RUN pip install 'cryptography>=43.0.1'

# TODO: remove rai-core-flask pin with next raiwidgets release
RUN pip install 'rai-core-flask==0.7.6'

# to resolve vulnerability mlflow (GHSA-wxj7-3fx5-pp9m) for mlflow < 3.1.0
# from azureml-mlflow package
RUN pip install 'mlflow>=3.1.0'

# To resolve vulnerability issue
RUN pip install 'gunicorn>=22.0.0'
RUN pip install 'Werkzeug>=3.0.3'
RUN pip install 'tqdm>=4.66.3'
# vulnerability of requests GHSA-9hjg-9r4m-mvj7
RUN pip install 'requests>=2.32.4'

# clean conda and pip caches
RUN conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

RUN rm -rf ~/.cache/pip

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $CONDA_PREFIX/lib:$LD_LIBRARY_PATH
