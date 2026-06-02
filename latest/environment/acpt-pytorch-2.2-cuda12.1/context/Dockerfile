FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.1

# Base conda env (Python 3.13) security floors. Tool-based parent checks on 2026-05-26
# showed requests only requires urllib3<3/idna<4, pydantic-settings only requires
# python-dotenv>=0.21.0, and click parents do not require a CVE-safe floor yet.
# pip is a root packaging tool and VCM also scans the ptca conda-meta record.
RUN conda install -n ptca -c conda-forge -y 'pip>=26.1' && \
    conda install -n base -c conda-forge -y 'pip>=26.1' 'conda=26.3.2' && \
    conda run -n base python -m pip install --upgrade --no-cache-dir \
        'python-dotenv>=1.2.2' \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'click>=8.3.3' && \
    conda clean -a -y && \
    rm -rf /opt/conda/pkgs

# Install pip dependencies (runs in default PTCA conda env)
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir


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
