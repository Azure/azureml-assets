FROM nvidia/cuda:11.3.1-devel-ubuntu18.04:{{latest-image-tag}}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    AML_APP_ROOT=/var/azureml-model/code BUILD_DIR=/tmp/build AZUREML_MODEL_DIR=/var/azureml-model \
    MII_MODEL_DIR=/var/azureml-model AZUREML_ENTRY_SCRIPT=score.py MII_CACHE_PATH=/tmp/mii_cache

COPY . $BUILD_DIR

RUN mkdir -p $BUILD_DIR && apt-get update && \
    apt-get install -y --no-install-recommends nginx-light wget sudo runit rsyslog libcurl3 unzip git-all && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /usr/share/man/* /var/lib/apt/lists/* && \
    mv "$BUILD_DIR/gunicorn_app" /etc/nginx/sites-available/ && \
    rm /etc/nginx/sites-enabled/default && \
    ln -s /etc/nginx/sites-available/gunicorn_app /etc/nginx/sites-enabled/ &&\
    mkdir -p /opt/miniconda /var/azureml-logger /var/azureml-util && cp -r "$BUILD_DIR/runit" /var && \
    mkdir -p {$AZUREML_MODEL_DIR,$MII_CACHE_PATH} && chmod 775 {$AZUREML_MODEL_DIR,$MII_CACHE_PATH}

ENV PATH=/opt/miniconda/envs/amlenv/bin:/opt/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    AZUREML_CONDA_ENVIRONMENT_PATH=/opt/miniconda/envs/amlenv \
    LD_LIBRARY_PATH=/usr/local/:/usr/local/lib:/usr/local/cuda:/usr/local/nvidia/lib:$LD_LIBRARY_PATH \
    SVDIR=/var/runit \
    AZUREML_INFERENCE_SERVER_HTTP_ENABLED=True

SHELL ["/bin/bash", "-c"]

RUN cd ~ && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -bf -p /opt/miniconda && \
    conda create -n amlenv python=3.8 -y

ENV PATH="/opt/miniconda/envs/amlenv/bin:$AML_APP_ROOT:$PATH" \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH="/opt/miniconda/envs/amlenv/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" \
    CONDA_DEFAULT_ENV=amlenv \
    PATH=$PATH:/usr/local/cuda/bin

RUN /opt/miniconda/envs/amlenv/bin/pip install -r "$BUILD_DIR/requirements.txt"

EXPOSE 5001

WORKDIR $AZUREML_MODEL_DIR/code
CMD sudo service nginx start && \
    cd $AZUREML_MODEL_DIR/code && azmlinfsrv --model_dir $AZUREML_MODEL_DIR --entry_script $AZUREML_MODEL_DIR/code/score.py --port 31311
