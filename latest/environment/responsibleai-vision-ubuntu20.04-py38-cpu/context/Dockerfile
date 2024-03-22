FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20240322.v1
# Install OpenCV native dependencies
RUN apt-get update
RUN apt-get install -y python3-opencv
RUN apt-get remove -y libavutil56 libswresample3 libavformat58
RUN apt-get remove -y libavcodec58 libswscale5 libopenexr24 libpmix2

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/responsibleai-vision

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

RUN conda list
# Conda install and pip install could happen side by side. Remove crypytography with vulnerability from conda
RUN conda remove cryptography

RUN pip install --pre 'azure-ai-ml' 'azure-storage-blob<=12.13.0'

# Install azureml packages
RUN pip install 'azureml-core==1.55.0.post2' \
                'azureml-mlflow==1.55.0' \
                'azureml-rai-utils==0.0.6'

RUN pip install 'azureml-dataset-runtime==1.55.0' \
                'azureml-automl-dnn-vision==1.55.0.post1'

# TODO: remove rai-core-flask pin with next raiwidgets release
RUN pip install 'shap==0.41.0' \
                'interpret-community==0.30.0' \
                'Pillow>=10.2.0' \
                'Werkzeug==2.2.3' \
                'setuptools>=65.5.1' \
                'numpy==1.22.0' \
                'scipy==1.10.0' \
                'statsmodels==0.14.0' \
                'rai-core-flask==0.7.4'

# azureml-dataset-runtime[fuse] upper bound for pyarrow is 11.0.0
# so we install pyarrow in extra step to avoid conflict
RUN pip install 'pyarrow>=14.0.1'

# To resolve vulnerability issue regarding crytography
RUN pip install 'cryptography>=42.0.4'

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
