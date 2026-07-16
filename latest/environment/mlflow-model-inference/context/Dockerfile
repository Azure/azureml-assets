FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20260715.v1

RUN mkdir -p /opt/miniconda/
RUN mkdir -p etc/nginx/sites-available

COPY runit/gunicorn /var/runit/gunicorn/
COPY runit/nginx /var/runit/nginx/
COPY runit/rsyslog /var/runit/rsyslog/
COPY common/aml_logger /var/azureml-logger
COPY utilities/start_logger.sh /var/azureml-logger/start_logger.sh
COPY configs/app etc/nginx/sites-available/app

RUN chmod +x var/runit/*/*
RUN chmod +x var/azureml-logger/start_logger.sh
#RUN chmod +x /var/runit/nginx/run

RUN ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
        rm -f /etc/nginx/sites-enabled/default
COPY configs/rsyslog.conf    etc/rsyslog.conf


RUN sed -i 's/\r$//g' /var/runit/gunicorn/run
RUN chmod +x /var/runit/gunicorn/run

RUN sed -i 's/\r$//g' /var/runit/gunicorn/finish
RUN chmod +x /var/runit/gunicorn/finish

RUN sed -i 's/\r$//g' /var/runit/nginx/run
RUN chmod +x /var/runit/nginx/run

RUN sed -i 's/\r$//g' /var/runit/nginx/finish
RUN chmod +x /var/runit/nginx/finish

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=300
ENV AZUREML_INFERENCE_SERVER_HTTP_ENABLED="True"

ENV PIP_USE_PEP517=1

EXPOSE 5001

COPY grant_ownership.sh /tmp/

RUN useradd --create-home dockeruser && \
    bash /tmp/grant_ownership.sh && rm -f /tmp/grant_ownership.sh

RUN chown -R dockeruser /var/runit
RUN chown -R dockeruser /var/log
RUN chown -R dockeruser /var/lib/nginx
RUN chown -R dockeruser /run
RUN chmod +x /var/azureml-logger/start_logger.sh
RUN chown -R dockeruser /var/azureml-app
RUN chown -R dockeruser:dockeruser /opt/miniconda

USER dockeruser
RUN /opt/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV PATH=/opt/miniconda/bin:$PATH

WORKDIR /
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/mlflow
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH

ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

ENV AML_APP_ROOT="/var/mlflow_resources"
ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

USER root

COPY mlmonitoring /var/mlflow_resources/mlmonitoring
COPY mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py
COPY mlflow_hf_score_cpu.py /var/mlflow_resources/mlflow_hf_score_cpu.py
COPY mlflow_hf_score_gpu.py /var/mlflow_resources/mlflow_hf_score_gpu.py

COPY conda_dependencies.yaml .
RUN conda clean -afy && \
    pip uninstall -y h2 || true && \
    conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y  

RUN pip install pipdeptree

USER dockeruser


CMD [ "runsvdir", "/var/runit" ]
