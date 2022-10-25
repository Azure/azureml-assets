
ARG BASE_IMAGE="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest"
FROM $BASE_IMAGE

ARG TEMP_ENV_YML="/tmp/env.yml"
COPY --chown=dockeruser env.yml $TEMP_ENV_YML

COPY mlflow_hf_score.py /var/mlflow_resources/mlflow_hf_score.py

ENV AML_APP_ROOT="/var/mlflow_resources"
ENV AZUREML_ENTRY_SCRIPT="mlflow_hf_score.py"

RUN conda init bash
RUN conda env create -f $TEMP_ENV_YML --force && \
        rm -rf $TEMP_ENV_YML && \
        CONDA_ROOT_DIR=$(conda info --root) && \
        rm -rf "$HOME/.cache/pip" && \
        conda clean -aqy && \
        rm -rf "$CONDA_ROOT_DIR/pkgs" && \
        find /opt/miniconda -type d -name __pycache__ -exec rm -rf {} +

USER root
COPY generate_notice.sh /
RUN bash generate_notice.sh && rm generate_notice.sh
USER dockeruser

CMD [ "runsvdir", "/var/runit" ]