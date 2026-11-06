FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.9
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV AZUREML_CONDA_ENVIRONMENT_PATH=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Keep Conda creation separate from pip resolution to avoid ACR build-agent OOMs.
COPY conda_dependencies.yaml .
RUN conda install -n base -c conda-forge --freeze-installed --force-reinstall -y pip=26.2.2 && \
    /opt/miniconda/bin/python -m pip install --no-cache-dir --force-reinstall pip==26.2.2 && \
    conda create -p $CONDA_PREFIX -q -y python=3.11 pip=26.2.2 pandas=2.2.3 'scipy~=1.10.0' numpy=1.26.4 -c conda-forge -c anaconda && \
    conda run -p $CONDA_PREFIX python -m pip install --no-cache-dir --force-reinstall pip==26.2.2 && \
    conda clean -a -y

RUN printf '%s\n' 'msgpack>=1.2.3' 'setuptools>=84.1.0' > /tmp/pip-security-constraints.txt && \
    conda run -p $CONDA_PREFIX pip install --no-cache-dir \
    --no-build-isolation \
    --constraint /tmp/pip-security-constraints.txt \
    scikit-learn-intelex=={{latest-pypi-version}} \
    azureml-core=={{latest-pypi-version}} \
    azureml-defaults=={{latest-pypi-version}} \
    azureml-inference-server-http=={{latest-pypi-version}} \
    azureml-mlflow=={{latest-pypi-version}} \
    azureml-telemetry=={{latest-pypi-version}} \
    'scikit-learn~=1.9.0' \
    'joblib~=1.6.0' \
    py-spy==0.3.12 \
    'debugpy~=1.8.0' \
    'ipykernel~=6.0' \
    tensorboard \
    'psutil~=5.9.4' \
    'matplotlib>=3.10.8,<3.11' \
    'tqdm~=4.66.3' \
    py-cpuinfo==5.0.0 \
    'torch-tb-profiler~=0.4.0' \
    'pillow>=13.0.0' \
    'starlette>=1.0.1' \
    'idna>=3.15' \
    'msgpack>=1.2.3' \
    'setuptools>=84.1.0' && \
    rm conda_dependencies.yaml /tmp/pip-security-constraints.txt

RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir --upgrade \
    'aiohttp>=3.14.4' 'pyOpenSSL==26.4.0' 'cryptography>=50.0.0,<51' 'distributed>=2026.1.0' 'GitPython>=3.1.55' 'msgpack>=1.2.3' 'pyarrow>=23.0.1,<24' 'setuptools>=84.1.0' 'sqlparse>=0.6.0'

RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade 'idna>=3.15' 'cryptography>=50.0.0'

RUN $CONDA_PREFIX/bin/python -m pip install --no-cache-dir --upgrade 'PyJWT>=2.15.0' 'oauthlib>=4.0.0' 'urllib3>=2.8.0' && \
    /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade 'PyJWT>=2.15.0' 'urllib3>=2.8.0' && rm -rf /opt/miniconda/pkgs/urllib3-2.7.0-*

RUN $CONDA_PREFIX/bin/python -m pip install --no-cache-dir --upgrade 'opentelemetry-api>=1.45.1'

RUN set -eux; \
    apt-get update; \
    security_packages="libbrotli1 nginx"; \
    installed_packages=""; \
    for package in $security_packages; do \
        if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then \
            installed_packages="$installed_packages $package"; \
        fi; \
    done; \
    if [ -n "$installed_packages" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade $installed_packages; \
    fi; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN $CONDA_PREFIX/bin/python -c \
    "import sys, sklearn, sklearnex, matplotlib, debugpy, psutil, tensorboard, torch_tb_profiler, azureml.core, azureml.mlflow, azureml_inference_server_http; \
    assert sys.version_info[:2] == (3, 11), sys.version; \
    assert sklearn.__version__.startswith('1.9.'), sklearn.__version__"
