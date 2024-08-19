# PTCA Image (enabling Singularity Support)
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch222:biweekly.202408.3

RUN apt-get update && apt-get -y upgrade

WORKDIR /
ENV CONDA_PREFIX=/azureml-envs/model-evaluation
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX

COPY conda.yaml .
# Create conda environment
RUN conda env create -p $CONDA_PREFIX -f conda.yaml -q && \
    rm conda.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y


# install transformers patch
COPY ./transformers.patch ./transformers.patch
RUN pip uninstall transformers -y && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout fc35907f95459d7a6c5281dfadd680b6f7b620e3 && \
    # git apply ../transformers.patch && \
    python3 setup.py bdist_wheel && \
    pip install dist/*.whl

COPY modeling_llama /opt/conda/envs/ptca/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

# Prepend path to AzureML conda environment
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $CONDA_PREFIX/lib:$LD_LIBRARY_PATH
