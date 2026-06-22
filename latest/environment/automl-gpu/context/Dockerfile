FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260614.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

ENV ENABLE_METADATA=true
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --only-upgrade \
        systemd \
        systemd-sysv \
        libudev1 \
        libpam-systemd \
        systemd-timesyncd \
        libsystemd0 \
        libnss-systemd \
        libpython3.10-stdlib \
        python3.10 \
        libpython3.10-minimal \
        python3.10-minimal \
        libpam0g \
        libpam-modules-bin \
        libpam-modules \
        libpam-runtime \
        libarchive13 \
        libgnutls30 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

# CVE-2026-41727: override cryptography to >=46.0.5,<47.0.0 (conda resolves older 44.x by default)
RUN pip install 'cryptography>=46.0.5,<47.0.0'

# CVE-2026-6357: upgrade pip in base miniconda (/opt/miniconda ships 26.0.1)
RUN /opt/miniconda/bin/python -m pip install --upgrade --no-cache-dir 'pip>=26.1.1' && \
    rm -rf /opt/miniconda/lib/python3.10/site-packages/pip-26.0.*.dist-info

# distributed/bokeh/onnx: parent packages cap these below patched versions.
# GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6,
# GHSA-538c-55jv-c5g9, GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m (onnx)
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --upgrade --no-cache-dir \
    'cryptography>=46.0.5,<47.0.0' 'setuptools>=79.0.0' 'distributed>=2026.1.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'pyarrow==23.0.1'  # pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.

RUN rm -rf /opt/miniconda/pkgs/

# Avoid ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AZUREML_CONDA_ENVIRONMENT_PATH/lib