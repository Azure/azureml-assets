FROM mcr.microsoft.com/azureml/inference-base-2204:{{latest-image-tag}}

WORKDIR /
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/mlflow
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

# Set MLFlow environment variables
ENV AML_APP_ROOT="/var/mlflow_resources"
ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

USER root

COPY mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

# Fix Trivy/Qualys vuln from pip's internally vendored setuptools (GHSA-5rjg-fvgr-3xxf, CVE-2025-47273).
# pip bundles a stale-versioned manifest of setuptools under pip/_vendor for its own internal use only;
# the real installed setuptools package is unaffected.
# conda clean -a -y above does not purge /opt/miniconda/pkgs for packages still owned by the base
# conda install (e.g. pip), so remove it explicitly first - otherwise the find below also patches the
# cached pip package's files, which breaks Conda's own integrity manifest for that cache entry and
# causes CondaVerificationError/SafetyError the next time an environment build re-links it from cache.
RUN rm -rf /opt/miniconda/pkgs && \
    find /opt/miniconda $AZUREML_CONDA_ENVIRONMENT_PATH -path '*/pip/_vendor/vendor.txt' \
        -exec sed -i '/^setuptools==/d' {} + && \
    find /opt/miniconda $AZUREML_CONDA_ENVIRONMENT_PATH -path '*/pip/_vendor/bom.cdx.json' -delete

USER dockeruser

CMD [ "runsvdir", "/var/runit" ]
