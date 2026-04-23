FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.6-ubuntu24.04:{{latest-image-tag}}

# OS updates + openssh (DeepSpeed multi-node launcher requires passwordless ssh)
# + inference-server scaffolding. This curated env is dual-use: its
# requirements.txt installs azureml-inference-server-http and inference-schema,
# so the nginx / runit / rsyslog / port layer below is part of the image's
# contract for online-endpoint deployments.
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        unzip \
        openssh-server \
        openssh-client && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*.deb && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# Upgrade pip toolchain before any package install so resolver sees patched versions
RUN pip install --no-cache-dir --upgrade \
        'pip>=26.0' \
        'setuptools>=82.0.1' \
        'wheel>=0.46.2'

# PyTorch 2.8 stack pinned to CUDA 12.6 wheels (base image ships CUDA 12.6.3)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.8.0' \
        'torchvision==0.23.0' \
        'torchaudio==2.8.0'

# DeepSpeed is required by pytorch2_8_sample_test.py (--deepspeed ds_config.json)
RUN pip install --no-cache-dir 'deepspeed>=0.15'

# AzureML integration + training toolchain. Security floors that upstream deps
# accept naturally (urllib3, pillow, aiohttp, filelock, requests) live in
# requirements.txt so pip resolves them in one pass.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force-upgrade only packages whose parent pins cap below the patched version:
#   - cryptography: azureml-mlflow~=1.62 caps <46
#   - protobuf:     azureml-mlflow / azureml-automl transitively cap <5
#   - PyJWT:        transitively capped below 2.12 by some azureml deps
RUN pip install --no-cache-dir --upgrade \
        'cryptography>=46.0.5' \
        'protobuf>=6.33.5' \
        'PyJWT>=2.12.0'

# Remove conda + pip caches to avoid opt/conda/pkgs/ scanner false-positives
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/ /root/.cache/pip
