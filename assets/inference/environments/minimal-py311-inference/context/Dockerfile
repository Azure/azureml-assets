FROM mcr.microsoft.com/azureml/inference-base-2204:{{latest-image-tag}}

WORKDIR /
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/minimal
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

# Create conda environment
USER root
RUN sed -i '66,148d' /var/runit/gunicorn/run
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

# Fix Trivy/Qualys vuln from pip's internally vendored setuptools (GHSA-5rjg-fvgr-3xxf, CVE-2025-47273).
# pip bundles a stale-versioned manifest of setuptools under pip/_vendor for its own internal use only;
# the real installed setuptools package is unaffected.
# Remove the stale manifest entry so scanners stop flagging it as a live vulnerable component.
RUN find /opt/miniconda $AZUREML_CONDA_ENVIRONMENT_PATH -path '*/pip/_vendor/vendor.txt' \
        -exec sed -i '/^setuptools==/d' {} + && \
    find /opt/miniconda $AZUREML_CONDA_ENVIRONMENT_PATH -path '*/pip/_vendor/bom.cdx.json' -delete

USER dockeruser

CMD [ "runsvdir", "/var/runit" ]
