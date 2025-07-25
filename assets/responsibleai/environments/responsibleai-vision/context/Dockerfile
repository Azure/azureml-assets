FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Install OpenCV native dependencies
RUN apt-get update
RUN apt-get install -y python3-opencv
RUN apt-get remove -y libavutil58 libswresample4 libavformat60
RUN apt-get remove -y libavcodec60 libswscale7 libpmix2 libcjson1
RUN apt-get remove -y liburiparser1 libzvbi0t64 libzvbi-common && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
ENV CONDA_PREFIX=/azureml-envs/responsibleai-vision
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml

# Prepend path to AzureML conda environment
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN conda list

RUN pip install --pre 'azure-ai-ml' 'azure-storage-blob<=12.13.0'

# Separating RAI install due to very old statsmodels dependency
# which conflicts with econml and numpy
RUN pip install 'responsibleai~=0.36.0' \
                'raiwidgets~=0.36.0' \
                'vision_explanation_methods' \
                'responsibleai-vision==0.3.8'

# Install azureml packages
RUN pip install 'azureml-rai-utils==0.0.6'

RUN pip install 'azureml-dataset-runtime=={{latest-pypi-version}}'

RUN pip install 'scikit-learn>=1.5.1' \
                'interpret-community==0.31.0' \
                'Pillow>=10.3.0' \
                'numpy==1.26.2' \
                'scipy==1.10.0' \
                'statsmodels==0.14.0' \
                'gunicorn>=22.0.0'

RUN pip install 'shap==0.46.0'

# azureml-dataset-runtime[fuse] upper bound for pyarrow is 11.0.0
# so we install pyarrow in extra step to avoid conflict
RUN pip install 'pyarrow>=14.0.1'

# To resolve vulnerability issue regarding crytography
RUN pip install 'cryptography>=43.0.1'

# To resolve vulnerability issue
RUN pip install 'Werkzeug>=3.0.3' \
                'tqdm>=4.66.3' \
                'onnx>=1.17.0'
# To resolve GHSA-2586-f3p4-hq84 vulnerability issue for lightgbm<4.6.0
# from azureml-automl-dnn-vision package
RUN pip install 'lightgbm>=4.6.0'

# To resolve GHSA-53q9-r3pm-6pq6 vulnerability issue for torch < 2.6.0
# and  urllib3 (GHSA-pq67-6m6q-mj2v) for urllib3 < 2.5.0
# from azureml-automl-dnn-vision package
RUN pip install 'torch>=2.7.1-rc1'
RUN pip install 'urllib3>=2.5.0'
# to resolve vulnerability mlflow (GHSA-wxj7-3fx5-pp9m) for mlflow < 3.1.0
# from azureml-mlflow package
RUN pip install 'mlflow>=3.1.0'

# To resolve GHSA-jh2j-j4j9-crg3 vulnerability issue
RUN pip install 'opencv-python-headless==4.8.1.78'
RUN pip install 'setuptools>=78.1.1'

# clean conda and pip caches
RUN conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

RUN rm -rf ~/.cache/pip

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $CONDA_PREFIX/lib:$LD_LIBRARY_PATH
