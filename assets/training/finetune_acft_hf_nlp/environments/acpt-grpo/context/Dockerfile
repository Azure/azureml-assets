#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
# Security: upgrade all OS packages to patch USN vulnerabilities (libssh, curl, openssl, etc.)
RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install transformers==4.56.0

# mlflow 3.5.0 has CVEs (CVE-2025-14287, CVE-2026-2033, CVE-2026-2635); upgrade after azureml packages
# azureml-evaluate-mlflow → azureml-mlflow pins mlflow-skinny<=3.5.0, conflicting with mlflow 3.10.1
# so mlflow must be upgraded separately to avoid pip resolution conflict
RUN pip install --no-cache-dir mlflow==3.10.1

# upgrade pip, wheel, setuptools and transitive deps to fix vulnerabilities
# protobuf: vllm uses loose floors, pip can't force 6.33.5 transitively
# cryptography: azureml-mlflow~=1.62.0 pins <46.0.0; override needed for CVE fix
RUN pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 protobuf==6.33.5 cryptography==46.0.6 xgrammar>=0.1.32 \
    'aiohttp>=3.13.4' 'onnx>=1.21.0' 'nltk>=3.9.4' 'pyasn1>=0.6.3' \
    'python-multipart>=0.0.22' 'pillow>=12.1.1' 'filelock>=3.20.3' 'urllib3>=2.6.3' 'ray>=2.54.0' \
    'azure-core>=1.38.0' 'cbor2>=5.9.0' 'PyJWT>=2.12.0' 'mlflow>=3.8.1,<4.0.0' 'vllm>=0.19.0'
# Fix additional transitive Python vulnerabilities not controlled by requirements.txt
# aiohttp: vllm uses loose floors, pip can't force >=3.13.3 transitively (GHSA-69f9 etc.)
# requests: GHSA-gc5v-m9x4-r6x2; PyJWT: CVE-2026-32597; azure-core: GHSA-jm66-cg57-jjv5
# cbor2: GHSA-3c37-wwvx-h642; pyasn1: CVE-2026-30922; python-multipart: GHSA-wp53-j4wj-2cfg
# urllib3: GHSA-38jv-5279-wg99
RUN pip install --no-cache-dir --upgrade 'aiohttp>=3.13.4' 'requests>=2.33.0' 'PyJWT>=2.12.0' \
    'azure-core>=1.38.0' 'cbor2>=5.9.0' 'pyasn1>=0.6.3' 'python-multipart>=0.0.22' 'urllib3>=2.6.3'
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y pip>=26.0.1 wheel>=0.46.2
# Fix vulnerabilities in base conda env (python3.13) from ACPT base image (biweekly.202603.1)
# Still vulnerable in base: cryptography(44.0.1), pip(25.3), setuptools(80.9.0), wheel(0.45.1)
RUN conda run -n base python -m pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.6
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
