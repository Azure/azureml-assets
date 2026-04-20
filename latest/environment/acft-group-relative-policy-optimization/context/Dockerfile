#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202604.1

USER root
# Security: upgrade all OS packages to patch USN vulnerabilities (libssh, curl, openssl, etc.)
RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow==0.0.89
RUN pip install azureml-acft-common-components==0.0.89
RUN pip install transformers==4.56.0

# mlflow 3.5.0 has CVEs (CVE-2025-14287, CVE-2026-2033, CVE-2026-2635); upgrade after azureml packages
# azureml-evaluate-mlflow → azureml-mlflow pins mlflow-skinny<=3.5.0, conflicting with mlflow 3.10.1
# so mlflow must be upgraded separately to avoid pip resolution conflict
RUN pip install --no-cache-dir mlflow==3.10.1

# upgrade pip, wheel, setuptools and transitive deps to fix vulnerabilities
# protobuf: vllm uses loose floors, pip can't force 6.33.5 transitively
# cryptography: azureml-mlflow~=1.62.0 pins <46.0.0; override needed for CVE fix
# onnx: onnxruntime-training accepts onnx>=1.16.0; override needed (GHSA-p433-9wv8-28xj etc.)
# nltk: GHSA-gfwx-w7gr-fvh7; >=3.9.4 required
# filelock: transitive dep of torch/huggingface-hub; parents use loose floor (GHSA-qmgc-5h2g-mvrw)
# urllib3: transitive dep of requests; parent uses urllib3>=1.21.1,<3 (GHSA-38jv-5279-wg99)
# ray: GHSA-q5fh-2hc8-f6rq; >=2.54.0 required
# azure-core: transitive dep of Azure SDKs; parents use loose floor (GHSA-jm66-cg57-jjv5)
# cbor2: transitive dep via azure-identity → msal-extensions; parent uses loose floor (GHSA-3c37-wwvx-h642)
# jaraco.context: transitive dep of keyring; parent uses loose floor (GHSA-58pv-8j8x-9vj2)
# mlflow: GHSA-r23q-823p-vmf7 etc.; azureml-mlflow pins mlflow-skinny<=3.5.0, override needed
# vllm: GHSA-pq5c-rjhq-qp7p, GHSA-3mwp-wvh9-7528 etc.; >=0.19.0 required
RUN pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 protobuf==6.33.5 cryptography==46.0.7 'xgrammar>=0.1.32' \
    'aiohttp>=3.13.4' 'requests>=2.33.0' 'onnx>=1.21.0' 'nltk>=3.9.4' 'pyasn1>=0.6.3' \
    'python-multipart>=0.0.22' 'pillow>=12.1.1' 'filelock>=3.20.3' 'urllib3>=2.6.3' 'ray>=2.54.0' \
    'azure-core>=1.38.0' 'cbor2>=5.9.0' 'jaraco.context>=6.1.0' 'PyJWT>=2.12.0' 'mlflow>=3.8.1,<4.0.0' 'vllm>=0.19.0'
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y 'pip>=26.0.1' 'wheel>=0.46.2'
# Fix vulnerabilities in base conda env (python3.13) from ACPT base image (biweekly.202603.1)
# Still vulnerable in base: cryptography(44.0.1), pip(25.3), setuptools(80.9.0), wheel(0.45.1), requests(2.32.4), aiohttp(3.12.14), PyJWT(2.10.1)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.7 \
    'requests>=2.33.0' 'aiohttp>=3.13.4' 'PyJWT>=2.12.0'
# ray vendors aiohttp for runtime_env agent under thirdparty_files; patch that copy too.
RUN rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/ray/__private/runtime_env/agent/thirdparty_files/aiohttp* && \
    pip install --no-cache-dir --target /opt/conda/envs/ptca/lib/python3.10/site-packages/ray/__private/runtime_env/agent/thirdparty_files 'aiohttp==3.13.4'
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
