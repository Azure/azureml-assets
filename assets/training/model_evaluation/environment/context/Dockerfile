FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get update && apt-get -y upgrade && \
    # libglib2.0-0 and libgl1 are needed by opencv-python, which MMDetection model wrappers install at load time.
    apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
# Keep this prefix: MLflow model wrappers (for example, MMDetection) reference it.
ENV CONDA_PREFIX=/azureml-envs/model-evaluation
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX

# Python 3.11 is required to load MLflow models saved by the Python 3.11 ACFT training images.
COPY conda.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda.yaml -q && \
    rm conda.yaml && \
    conda clean -a -y

ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install CUDA 12.6 PyTorch wheels to match the base image runtime.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.13.0+cu126' 'torchvision==0.28.0+cu126'

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Temp fix for env vuln
RUN pip install --no-cache-dir mlflow=={{latest-pypi-version}} transformers=={{latest-pypi-version}}

# MMDetection MLflow model wrappers run `mim install` (openmim 0.3.9) when loading the model.
# Newer pip versions make mim skip mmcv, and the runtime installs can pull numpy 2, which breaks pandas.
RUN pip install --no-cache-dir pip==24.0 && echo 'numpy<2' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN rm -rf ~/.cache/pip
