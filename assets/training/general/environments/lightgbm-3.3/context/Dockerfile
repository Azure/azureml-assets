FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/lightgbm-3.3
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Update all system packages to pick up available security patches
RUN apt-get update && apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# TODO: Patch needed for Ubuntu Security Notification for libxml2 Vulnerabilities (USN-7974-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for GNU C Library Vulnerabilities (USN-8005-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for Python Vulnerabilities (USN-8018-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for GLib Vulnerabilities (USN-8017-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for nginx Vulnerability (USN-8038-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for GnuTLS Vulnerabilities (USN-8043-1) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for Expat Vulnerabilities (USN-8022-2) - awaiting patched version
# TODO: Patch needed for Ubuntu Security Notification for curl Vulnerabilities (USN-8062-1) - awaiting patched version

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# Security vulnerability fixes
RUN conda run -p $CONDA_PREFIX pip install --upgrade \
    urllib3==2.6.3 \
    aiohttp==3.13.3 \
    PyNaCl==1.6.2 \
    # GHSA-r6ph-v2qm-q3c2 - cryptography missing EC subgroup validation (CVE-2026-26007)
    'cryptography>=46.0.5' \
    # GHSA-cfh3-3jmp-rvhc - pillow OOB write on PSD images (CVE-2026-25990)
    'pillow>=12.1.1' \
    # GHSA-7gcm-g887-7qv7 - protobuf JSON recursion depth bypass DoS (CVE-2026-0994)
    'protobuf>=6.33.5' \
    # GHSA-8rrh-rw8j-w5fx - wheel path traversal in unpack (CVE-2026-24049)
    'wheel>=0.46.2' \
    # GHSA-6vgw-5pg2-w6jp - pip path traversal during wheel extraction (CVE-2026-1703)
    'pip>=26.0' \
    # GHSA-c336-7962-wfj2 - distributed RCE via XSS (CVE-2026-23528)
    'distributed>=2026.1.0'
