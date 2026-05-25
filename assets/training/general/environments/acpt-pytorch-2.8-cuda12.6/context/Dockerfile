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

# Deepspeed launcher requires passwordless ssh login
RUN apt-get update && \
    apt-get install -y openssh-server openssh-client && \
    apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix security vulnerabilities in ptca conda env (python 3.10, active)
# setuptools>=82.0.1: GHSA-58pv-8j8x-9vj2, GHSA-8rrh-rw8j-w5fx
# urllib3>=2.7.0: GHSA-mf9v-mfxr-j63j (CVE-2026-44432), GHSA-qccp-gfcp-xxvc
#   (CVE-2026-44431). Transitive via requests<=2.34.2 (urllib3<3,>=1.26) and
#   botocore<=1.43.14 (urllib3!=2.2.0,<3,>=1.25.4); neither parent pins a safe
#   floor as of 2026-05-23, so direct override required.
RUN pip install --upgrade 'setuptools>=82.0.1' 'urllib3>=2.7.0'

# pip>=26.1: CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9). pip is a root tool with no
# parent package. Scanner reads opt/conda/envs/ptca/conda-meta/pip-*.json which
# is only updated by conda, so we must use `conda install` (a plain
# `pip install --upgrade pip` leaves the conda-meta record stale).
RUN conda install -n ptca -c conda-forge -y 'pip>=26.1'

# Fix security vulnerabilities in base conda env (python 3.13)
# setuptools>=82.0.1: same advisories as ptca env above.
# python-dotenv>=1.2.2: CVE-2026-28684 (GHSA-mf9w-mj56-hr94). Transitive via
#   azureml-defaults -> azureml-inference-server-http -> pydantic-settings
#   (<=2.14.0 requires python-dotenv>=0.21.0, no safe floor); direct override
#   required.
# urllib3>=2.7.0: see ptca env block above; same parent analysis.
# idna>=3.15: GHSA-65pc-fj4g-8rjx (CVE-2026-45409). Transitive via
#   requests<=2.34.2 (idna<4,>=2.5); parent does not pin a safe floor as of
#   2026-05-23, so direct override required.
# click>=8.3.3: GHSA-47fr-3ffg-hgmw (CVE-2026-7246). Transitive via
#   mlflow-skinny<=3.12.0 (click<9,>=7.0), flask<=3.1.3 (click>=8.1.3),
#   uvicorn<=0.47.0 (click>=7.0), typer<=0.25.1 (click>=8.2.1); none pin a
#   safe floor as of 2026-05-23, so direct override required.
RUN conda run -n base python -m pip install --upgrade \
    'setuptools>=82.0.1' 'python-dotenv>=1.2.2' \
    'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3'

# pip>=26.1 in base env: CVE-2026-6357 (GHSA-jp4c-xjxw-mgf9). Same reasoning as
# ptca env: conda-managed pip needs `conda install` so conda-meta and the
# matching dist-info are updated together.
RUN conda install -n base -c conda-forge -y 'pip>=26.1'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
