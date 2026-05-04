FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN apt-get update && \
    apt-get upgrade -y && \
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
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# support Deepspeed launcher requirement of passwordless ssh login
# Remove dotnet-* packages once its fixed in base image (04-05-2026)
RUN apt-get update && \
    apt-get install -y openssh-server openssh-client && \
    apt-get upgrade -y && \
    apt-get install -y --only-upgrade \
        dotnet-host-8.0 \
        dotnet-hostfxr-8.0 \
        dotnet-runtime-8.0 || true && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Fix security vulnerabilities in ptca conda env (active environment)
# pip>=26.0: GHSA-4xh5-x5gv-qwph, GHSA-6vgw-5pg2-w6jp
# wheel>=0.46.2: CVE-2026-24049
# cryptography>=46.0.7: CVE-2026-41727 (NOTE: azureml-mlflow~=1.62.0 pins cryptography<47.0.0; compatible)
# pillow>=12.1.1: GHSA-cfh3-3jmp-rvhc; transitive dep of matplotlib (requires >=8), no parent upgrade resolves this
# protobuf>=6.33.5: CVE-2026-40186; transitive dep of onnxruntime/tensorboard, parents use loose floors
# setuptools>=82.0.1: GHSA-58pv-8j8x-9vj2 (vendored jaraco.context), GHSA-8rrh-rw8j-w5fx (vendored wheel)
# pytest>=9.0.3: GHSA-6w46-j5rx-g56g; pre-installed in ptca env from ACPT base image (7.4.3), no parent to upgrade
# GitPython>=3.1.47: GHSA-x2qx-6953-8485, GHSA-rpm5-65cw-6hj4; transitive dep chain: azureml-mlflow -> mlflow-skinny
#   -> GitPython (>=3.1.9,<4). mlflow-skinny (all versions through 3.11.1) uses loose floor, no parent upgrade resolves this
RUN pip install --upgrade 'pip>=26.0' 'wheel>=0.46.2' 'cryptography>=46.0.7' 'pillow>=12.1.1' 'protobuf>=6.33.5' 'setuptools>=82.0.1' 'pytest>=9.0.3' 'GitPython>=3.1.47'

# Fix security vulnerabilities in base conda env (python 3.13)
# Same CVEs as above for pip/wheel/cryptography/pillow/protobuf/setuptools, plus:
# PyJWT>=2.12.0: CVE-2026-32597; transitive dep of msal (>=1.0.0,<3) and azureml-core (<3.0.0), no parent pins safe floor
# aiohttp>=3.13.4: GHSA-63hf-3vf5-4wqf et al. (8 CVEs); transitive dep of azure SDK, parents use loose floors
# python-dotenv>=1.2.2: CVE-2026-28684 (GHSA-mf9w-mj56-hr94); transitive dep chain: azureml-defaults
#   -> azureml-inference-server-http -> pydantic-settings -> python-dotenv>=0.21.0.
#   pydantic-settings (all versions through 2.14.0) only requires >=0.21.0, no parent upgrade resolves this
# requests>=2.33.0: GHSA-gc5v-m9x4-r6x2; transitive dep of msal (>=2.0.0,<3) and azureml-core (>=2.19.1,<3), no parent pins safe floor
# urllib3>=2.6.3: CVE-2026-37152; transitive dep of requests/azure SDK, parents use loose floors
RUN conda run -n base python -m pip install --upgrade 'pip>=26.0' 'wheel>=0.46.2' 'cryptography>=46.0.7' 'pillow>=12.1.1' 'protobuf>=6.33.5' 'setuptools>=82.0.1' 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'python-dotenv>=1.2.2' 'requests>=2.33.0' 'urllib3>=2.6.3'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -y -n ptca pip>=26.0.1
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/