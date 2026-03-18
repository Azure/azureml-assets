#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install transformers==4.56.0

# upgrade pip, wheel, setuptools and transitive deps to fix vulnerabilities
# protobuf is a transitive dep of vllm; parent uses loose floors, cannot force 6.33.5
# aiohttp is a transitive dep of vllm/ray; jaraco.context is a transitive dep of setuptools
# xgrammar is a transitive dep of vllm; ray is a transitive dep of vllm
RUN pip install --upgrade pip==26.0 wheel==0.46.2 setuptools==82.0.0 protobuf==6.33.5 cryptography==46.0.5 pyasn1==0.6.2 python-multipart==0.0.22 urllib3==2.6.3 azure-core==1.38.0 \
    aiohttp==3.13.3 jaraco.context==6.1.0 xgrammar>=0.1.32 ray>=2.54.0

# Patch Ray-bundled Java dependencies (log4j-core CVE-2025-68161, jackson-core GHSA-72hv-8253-57qq)
RUN rm -f $(python -c "import ray;print(ray.__path__[0])")/jars/ray_dist.jar && \
    rm -rf $(python -c "import ray;print(ray.__path__[0])")/_private/runtime_env/agent/thirdparty_files/*aiohttp*

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

# vulnerability in base conda env
RUN conda run -n base python -m pip install --upgrade urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 pip==26.0 wheel==0.46.2 setuptools==82.0.0 cryptography==46.0.5

# TODO: Patch needed for Ubuntu Security Notification for GnuPG Vulnerability (USN-7946-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for Python Vulnerability (USN-7951-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for Libtasn1 Vulnerabilities (USN-7954-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for libpng Vulnerabilities (USN-7963-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for libxml2 Vulnerabilities (USN-7974-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for OpenSSL Vulnerabilities (USN-7980-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for GNU C Library Vulnerabilities (USN-8005-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for Python Vulnerabilities (USN-8018-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for Expat Vulnerabilities (USN-8022-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for .NET Vulnerability (USN-8025-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for libpng Vulnerability (USN-8039-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for GnuTLS Vulnerabilities (USN-8043-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for alsa-lib Vulnerability (USN-8044-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for curl Vulnerabilities (USN-8062-1) - awaiting patched version
