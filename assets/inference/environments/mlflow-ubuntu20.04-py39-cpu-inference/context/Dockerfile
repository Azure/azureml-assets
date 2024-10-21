FROM mcr.microsoft.com/azureml/inference-base-2004:{{latest-image-tag}}
# ARG BASE_IMAGE
# FROM $BASE_IMAGE
WORKDIR /
ARG IMAGE_INFORMATION=None
ARG RESOURCES_DIR
ARG TEMP_ENV_YML="/tmp/env.yml"

COPY --chown=dockeruser $RESOURCES_DIR/environment.yaml $TEMP_ENV_YML

# Commenting the copying mlmonitoring step, will add once testings are completed.
# COPY --chown=dockeruser $RESOURCES_DIR/context/mlmonitoring /var/mlflow_resources/mlmonitoring

# We'll copy the HF scripts as well to enable better handling for v2 packaging. This will not require changes to the
# packages installed in the image, as the expectation is that these will all be brought along with the model.
COPY $RESOURCES_DIR/context/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py
COPY $RESOURCES_DIR/context/mlflow_hf_score_cpu.py /var/mlflow_resources/mlflow_hf_score_cpu.py
COPY $RESOURCES_DIR/context/mlflow_hf_score_gpu.py /var/mlflow_resources/mlflow_hf_score_gpu.py

ENV AML_APP_ROOT="/var/mlflow_resources"
ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

RUN conda init bash
RUN conda env create -f $TEMP_ENV_YML && \
        rm -rf $TEMP_ENV_YML && \
        CONDA_ROOT_DIR=$(conda info --root) && \
        rm -rf "$HOME/.cache/pip" && \
        conda clean -aqy && \
        rm -rf "$CONDA_ROOT_DIR/pkgs" && \
        find /opt/miniconda -type d -name __pycache__ -exec rm -rf {} +

USER root
COPY generate_notice.sh /
RUN bash generate_notice.sh && rm generate_notice.sh
RUN echo $IMAGE_INFORMATION > /IMAGE_INFORMATION
USER dockeruser

CMD [ "runsvdir", "/var/runit" ]