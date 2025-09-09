FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202509.1

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Upgrade known vulnerable packages again just to be safe
RUN pip install --upgrade \
    requests==2.32.4 \
    urllib3==2.5.0 \
    pillow==11.3.0 || true

# Repeat for other envs if applicable
RUN /opt/conda/bin/pip install --upgrade \
    requests==2.32.4 \
    urllib3==2.5.0 \
    pillow==11.3.0 || true

RUN /opt/conda/envs/ptca/bin/pip install --upgrade \
    requests==2.32.4 \
    urllib3==2.5.0 \
    pillow==11.3.0 || true

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
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

RUN apt-get update && \
    apt-get install -y --only-upgrade \
        libpython3.10-stdlib \
        python3.10 \
        libpython3.10-minimal \
        python3.10-minimal \
        libpam0g \
        libpam-modules-bin \
        libpam-modules \
        libpam-runtime \
        sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client
