FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202512.1

# Upgrade base conda packages
RUN conda install -n base -c conda-forge pip=25.3 -y
RUN conda list -n base

# Upgrade ptca environment conda packages
RUN conda install -c conda-forge pip=25.3 -y && \
    conda clean -a -y

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
RUN conda list

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
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        dpkg \
        dpkg-dev \
        libbinutils \
        binutils \
        binutils-common \
        binutils-x86-64-linux-gnu \
        libctf0 \
        libctf-nobfd0 \
        libc6 \
        libc6-dev \
        libc-bin \
        libc-dev-bin \
        libdpkg-perl \
        libssh-4 \
        libxml2 \
        linux-libc-dev \
        linux-headers-generic \
        locales \
        libssl-dev \
        libssl3 \
        openssl \
        sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y 

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client
