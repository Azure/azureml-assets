#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --only-upgrade linux-headers-5.15.0-163-generic=5.15.0-163.173 linux-headers-5.15.0-163=5.15.0-163.173 linux-libc-dev=5.15.0-163.173 || \
    apt-get install -y --only-upgrade linux-headers-generic linux-libc-dev && \
    apt-get install -y --only-upgrade git tar binutils binutils-common binutils-x86-64-linux-gnu libbinutils wget && \
    apt-get autoremove -y linux-headers-5.15.0-153 linux-headers-5.15.0-153-generic linux-headers-5.15.0-161 linux-headers-5.15.0-161-generic 2>/dev/null && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN conda run -n base python -m pip install --upgrade pip==26.0 urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.5
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}

# following are for vulnerability overrides at later\
# release of following packages consider moving then to requirements.txt
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
RUN pip install --no-cache-dir --upgrade "wandb>=0.23.0"
# GHSA-7rgv-gqhr-fxg3: xgrammar stack exhaustion DoS; patched in 0.1.32
RUN pip install xgrammar==0.1.32
RUN pip install transformers==4.57.1
# upgrade pip, wheel, setuptools to fix vulnerabilities
# jaraco.context>=6.1.0: CVE-2026-23949 Zip Slip path traversal fix. Direct install required because
# setuptools 82.0.1 (latest) vendors jaraco.context with vulnerable code despite dist-info labeling 6.1.0
RUN pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.5 protobuf==6.33.5 pillow==12.1.1 jaraco.context==6.1.0 \
    'aiohttp>=3.13.3' 'azure-core>=1.38.0' 'pyasn1>=0.6.3' 'urllib3>=2.6.3' 'python-multipart>=0.0.22' 'tornado>=6.5.5' 'filelock>=3.20.3' 'orjson>=3.11.6'
# Fix vulnerabilities in the ptca conda environment (pre-built in base image, not targeted by above installs)
# CVE-2026-1703 (pip), CVE-2026-24049 (wheel), CVE-2026-32597 (PyJWT via azureml-core)
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade "pip>=26.0" "wheel>=0.46.2" "PyJWT>=2.12.0"
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
