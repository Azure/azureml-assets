FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.2.v1

# Base conda env (Python 3.13) security floors.
# PyJWT>=2.13.0: pinned transitive dep brought in by the base image; fixes GHSA-jq35-7prp-9v3f and GHSA-993g-76c3-p5m4.
# py-rattler>=0.24.0: pinned transitive dep of base conda tooling; brought in by base image, fixes GHSA-q53q-5r4j-5729.
# cryptography>=48.0.1: pinned transitive dep brought in by the base image; fixes GHSA-537c-gmf6-5ccf.
RUN conda install -n ptca -c conda-forge -y 'pip>=26.1' && \
    conda install -n base -c conda-forge -y 'pip>=26.1' 'conda=26.3.2' && \
    conda run -n base python -m pip install --upgrade --no-cache-dir \
        'python-dotenv>=1.2.2' \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'click>=8.3.3' \
        'aiohttp>=3.14.0' \
        'PyJWT>=2.13.0' \
        'py-rattler>=0.24.0' \
        'cryptography>=48.0.1' && \
    conda run -n ptca python -m pip install --upgrade --no-cache-dir \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'cryptography>=48.0.1' && \
    conda clean -a -y && \
    rm -rf /opt/conda/pkgs

# Install pip dependencies (runs in default PTCA conda env)
COPY requirements.txt .
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
RUN pip install -r requirements.txt --no-cache-dir && \
    pip install --upgrade --no-cache-dir --no-deps 'pyarrow>=23.0.1'

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# Deepspeed launcher requires passwordless ssh; server side not in base image
RUN apt-get update && apt-get install -y openssh-server && rm -rf /var/lib/apt/lists/*
